r"""
Test 157: Tier 5 (graph) — BREADTH COUNT-WITH-CONDITION over page-only bridge main spans.
Level: graph   Weight: long   Difficulty: 8/10

SEVEN named suspension bridges, NO URLs. For each bridge the agent must open that bridge's own
Wikipedia article and read ONE figure from its infobox — the LONGEST (main) SPAN in metres —
then COUNT how many of the seven have a main span STRICTLY GREATER THAN 1,200 m.

The seven lookups are completely independent: no bridge's span depends on any other bridge, so
a graph can dispatch all seven arms in ONE parallel wave and only the terminal count needs the
gathered set. A linear ReAct agent must serialize seven gather-hops into a capped step budget
and hold seven four-digit values in one degrading scratchpad before applying the threshold.
That is the breadth axis this task exists to measure (sibling shape of tests 072/152).

Ground truth — every value verified against the LIVE English Wikipedia infobox 'Longest span'
field on 2026-08-22 (one WebFetch per article, exact figure confirmed as printed):

  bridge                       country        article title                      span     >1200 m?
  ─────────────────────────────────────────────────────────────────────────────────────────────
  Xihoumen Bridge              China          Xihoumen Bridge                    1,650 m    YES
  Yi Sun-sin Bridge            South Korea    Yi Sun-sin Bridge                  1,545 m    YES
  Yavuz Sultan Selim Bridge    Turkey         Yavuz Sultan Selim Bridge          1,408 m    YES
  Jiangyin Yangtze River Br.   China          Jiangyin Yangtze River Bridge      1,385 m    YES
  Ōnaruto Bridge               Japan          Ōnaruto Bridge                       876 m    NO
  Askøy Bridge                 Norway         Askøy Bridge                         850 m    NO
  Angostura Bridge             Venezuela      Angostura Bridge                     712 m    NO
  ─────────────────────────────────────────────────────────────────────────────────────────────
  KEYSTONE = count of bridges with main span > 1,200 m = 4   (4 of 7 — mid-range, not 0/6/7)

THRESHOLD MARGINS (every span clears or misses 1,200 m by >= 185 m, i.e. >= 15% of the
threshold — no borderline item):
  PASS  Xihoumen                1650 − 1200 = +450 m
  PASS  Yi Sun-sin              1545 − 1200 = +345 m
  PASS  Yavuz Sultan Selim      1408 − 1200 = +208 m
  PASS  Jiangyin                1385 − 1200 = +185 m   (tightest pass margin)
  FAIL  Ōnaruto                  876 − 1200 = −324 m   (tightest fail margin)
  FAIL  Askøy                    850 − 1200 = −350 m
  FAIL  Angostura                712 − 1200 = −488 m
  A ±100 m misread of a four-digit infobox figure cannot flip ANY item's pass/fail status, so
  the count is robust to a noisy extraction on any single arm.

ANTI-PARAMETRIC: the famous long-span bridges (Akashi Kaikyō, Golden Gate, Humber, 1915
Çanakkale) are deliberately EXCLUDED. The seven chosen bridges span six countries and are
solid-but-non-iconic; their exact spans are page-specific numbers that are not simultaneously
recallable, and the COUNT (4) is not a published fact anywhere. Sibling test 150 uses a bridge
span too, but a DIFFERENT bridge (Hardanger, 1,310 m) and a different shape (race-and-merge);
no fixture is shared between the two tasks.

DECOY NOTE: each of these infoboxes prints a TOTAL LENGTH as well as the longest span (e.g.
Xihoumen's deck is far longer than its 1,650 m main span). Grabbing the wrong infobox row is
the natural failure mode, which is why every leaf instruction names the 'Longest span' field
explicitly and forbids total length.

KEYSTONE (hard 0/1)   = the integer 4 in the primary answer, AND grounding (visit.count > 0).
COVERAGE   (un-gated) = how many of the seven (bridge, span) pairs were gathered.
CLASSIFY   (un-gated) = how many of the seven were correctly labelled above/below threshold —
                        the per-item breadth diagnostic, scored even when the final count is
                        botched (a linear agent that drops arms loses here regardless).
PASSING/CITATION      = gated secondaries; both short-circuit to 0 without the keystone.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ── verified fixtures ────────────────────────────────────────────────────────────────────────────
# 'span' = longest (main) span in metres from the Wikipedia infobox 'Longest span' field.
# 'passes' = (span > THRESHOLD). Nothing below is leaked into the task statement or the plan.

THRESHOLD: int = 1200  # metres, strictly greater-than

ENTITIES: List[Dict[str, Any]] = [
    {
        "key":      "xihoumen",
        "name":     "Xihoumen Bridge",
        "country":  "China (Zhejiang)",
        "span":     1650,
        "passes":   True,
        "name_rx":  r"xihoumen",
        "span_rx":  r"(?<!\d)1[,\s]?650(?!\d)",
        "slug_rx":  r"wiki/xihoumen_bridge",
    },
    {
        "key":      "yi_sun_sin",
        "name":     "Yi Sun-sin Bridge",
        "country":  "South Korea (Yeosu)",
        "span":     1545,
        "passes":   True,
        "name_rx":  r"yi\s*sun[\s\-]?sin",
        "span_rx":  r"(?<!\d)1[,\s]?545(?!\d)",
        "slug_rx":  r"wiki/yi[_%20\-]?sun[_%20\-]?sin[_%20\-]?bridge",
    },
    {
        "key":      "yavuz_sultan_selim",
        "name":     "Yavuz Sultan Selim Bridge",
        "country":  "Turkey (Istanbul)",
        "span":     1408,
        "passes":   True,
        "name_rx":  r"yavuz|sultan\s+selim",
        "span_rx":  r"(?<!\d)1[,\s]?408(?!\d)",
        "slug_rx":  r"wiki/yavuz[_%20]?sultan[_%20]?selim[_%20]?bridge",
    },
    {
        "key":      "jiangyin",
        "name":     "Jiangyin Yangtze River Bridge",
        "country":  "China (Jiangsu)",
        "span":     1385,
        "passes":   True,
        "name_rx":  r"jiangyin",
        "span_rx":  r"(?<!\d)1[,\s]?385(?!\d)",
        "slug_rx":  r"wiki/jiangyin[_%20a-z]*bridge",
    },
    {
        "key":      "onaruto",
        # Article title is "Ōnaruto Bridge"; the URL percent-encodes Ō as %C5%8C.
        "name":     "Ōnaruto Bridge",
        "country":  "Japan (Tokushima/Hyōgo)",
        "span":     876,
        "passes":   False,
        "name_rx":  r"[oō]naruto",
        "span_rx":  r"(?<!\d)876(?!\d)",
        "slug_rx":  r"wiki/(%c5%8c|o|ō)naruto[_%20]?bridge",
    },
    {
        "key":      "askoy",
        # Article title is "Askøy Bridge"; the URL percent-encodes ø as %C3%B8.
        "name":     "Askøy Bridge",
        "country":  "Norway (Vestland)",
        "span":     850,
        "passes":   False,
        "name_rx":  r"ask(ø|o|oe|%c3%b8)y",
        "slug_rx":  r"wiki/ask(%c3%b8|ø|o|oe)y[_%20]?bridge",
        "span_rx":  r"(?<!\d)850(?!\d)",
    },
    {
        "key":      "angostura",
        "name":     "Angostura Bridge",
        "country":  "Venezuela (Ciudad Bolívar)",
        "span":     712,
        "passes":   False,
        "name_rx":  r"angostura",
        "span_rx":  r"(?<!\d)712(?!\d)",
        "slug_rx":  r"wiki/angostura[_%20]?bridge",
    },
]

PASSING: List[Dict[str, Any]] = [e for e in ENTITIES if e["passes"]]        # 4 bridges
FAILING: List[Dict[str, Any]] = [e for e in ENTITIES if not e["passes"]]    # 3 bridges
KEYSTONE_COUNT: int = len(PASSING)                                         # 4

# Numeric-token extractor: maximal digit runs (with internal grouping commas). ``_int_values``
# then keeps only PLAIN integers (decimal-point tokens are dropped).
_NUM_TOKEN_RX = re.compile(r"\d[\d.,]*\d|\d")

# Per-item verdict cues for the un-gated classification diagnostic. Scanned inside a
# period-bounded, NEWLINE-TOLERANT window ([^.]) after/before the bridge name; the EARLIEST
# match in the window wins, so (a) "does not exceed" beats the bare "exceed" that starts later
# inside it, and (b) a verdict belonging to the NEXT row cannot outrank this row's own verdict.
_ABOVE_RX = (
    r"\bexceed(?:s|ed|ing)?\b|\babove\b|\bover\b|\bgreater\b|\blonger\s+than\b|\bmore\s+than\b"
    r"|✓|\byes\b|\bpass(?:es|ed)?\b|\btrue\b"
)
_BELOW_RX = (
    r"\b(?:do(?:es)?|did|dos)?\s*n[o']t\s+(?:exceed\w*|reach|go\s+above)\b"
    r"|\bnot\s+(?:exceed\w*|above|greater|longer|more)\b"
    r"|\bbelow\b|\bunder\b|\bshorter\s+than\b|\bless\s+than\b|✗|✘"
    r"|\bno\b|\bfail(?:s|ed)?\b|\bfalse\b"
)
_VERDICT_RX = re.compile(rf"(?P<below>{_BELOW_RX})|(?P<above>{_ABOVE_RX})", re.IGNORECASE)


# ── metadata ─────────────────────────────────────────────────────────────────────────────────────

def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id":          "157",
        "test_name":        "Tier 5: Count-with-condition — bridges with main span > 1,200 m (7-way fan-out)",
        "difficulty_level": "8/10",
        "category":         "Count-with-condition (page-only bridge main spans, 7-way fan-out)",
        "level":            "graph",
        "weight":           "long",
    }


# ── task statement ───────────────────────────────────────────────────────────────────────────────

def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {e['name']} — {e['country']}" for i, e in enumerate(ENTITIES, 1))
    return (
        "You are given NO URLs — search to find the pages you need, then READ them (do not "
        "guess from memory). For EACH of the following seven bridges, open the bridge's own "
        "Wikipedia article and read ONE figure directly from its infobox: the LONGEST SPAN "
        "(the main span) in metres — the 'Longest span' field. This is NOT the bridge's total "
        "length and NOT its height or clearance:\n"
        f"{listing}\n\n"
        "The seven lookups are completely independent of one another.\n\n"
        f"Then COUNT how many of the seven bridges have a longest span STRICTLY GREATER THAN "
        f"{THRESHOLD:,} metres, and report:\n"
        f"  (a) the count — how many of the seven exceed {THRESHOLD:,} m (a single integer 0–7; "
        "this is the keystone answer and belongs in your primary answer),\n"
        f"  (b) which specific bridges exceed {THRESHOLD:,} m and which do not (two named lists),\n"
        "  (c) each bridge's longest span in metres as read from its infobox, and\n"
        "  (d) the exact source URL of every Wikipedia page you read.\n\n"
        "IMPORTANT: do not guess spans from memory or from a bridge's fame. You MUST open each "
        "bridge's Wikipedia page and read the infobox 'Longest span' figure."
    )


# ── deliverables / success criteria ──────────────────────────────────────────────────────────────

def get_required_deliverables() -> List[str]:
    return [
        f"Count of bridges with a longest span > {THRESHOLD:,} m (the keystone integer, 0–7)",
        f"Which bridges exceed {THRESHOLD:,} m and which do not (passing and failing names)",
        "Each bridge's longest span in metres (seven individual infobox values)",
        "Source URL for each bridge's Wikipedia page",
    ]


def get_success_criteria() -> List[str]:
    passing_names = ", ".join(e["name"] for e in PASSING)
    failing_names = ", ".join(e["name"] for e in FAILING)
    return [
        f"At least 6 pages visited (target {len(ENTITIES)}, one per bridge)",
        f"Correctly reports the count as {KEYSTONE_COUNT} "
        f"(the number of bridges with a longest span > {THRESHOLD:,} m)",
        f"Identifies the {len(PASSING)} passing bridges: {passing_names}",
        f"Identifies the {len(FAILING)} failing bridges: {failing_names}",
        "Reports all seven longest-span figures from their infoboxes",
        "Cites a Wikipedia source URL for each bridge",
    ]


# ── shared text helpers ──────────────────────────────────────────────────────────────────────────

def _primary_text(result: Dict[str, Any]) -> str:
    """Primary answer text: deliverables[0] if populated, else the final deliverable."""
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list) and deliv and deliv[0] is not None:
            return str(deliv[0])
    return extract_final_text(result)


def _all_text(result: Dict[str, Any]) -> str:
    """Full reported text: final_deliverable plus every deliverable slot concatenated."""
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return " ".join(parts)


def _int_values(text: str) -> List[int]:
    """Plain integer values in ``text`` (decimal tokens dropped, grouping commas stripped)."""
    vals: List[int] = []
    for tok in _NUM_TOKEN_RX.findall(text):
        if "." in tok:
            continue
        try:
            vals.append(int(tok.replace(",", "")))
        except ValueError:
            continue
    return vals


_LIST_MARKER_RX = re.compile(r"(?m)^\s*\(?(\d{1,2})[.)]\s+")


def _strip_list_markers(text: str) -> str:
    """Drop a leading enumeration marker ('1. ', '2) ', '(3) ') from each line before reading
    asserted values — but ONLY when the text really looks like a numbered list (>= 2 markers).
    A single leading digit-marker on a terse answer ('4.' / '4. bridges exceed the threshold')
    is far more likely the genuine answer than list enumeration, so it is left alone. Mirrors
    the hardened helper in test 072."""
    if len(_LIST_MARKER_RX.findall(text)) < 2:
        return text
    return _LIST_MARKER_RX.sub("", text)


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE gate: the primary answer slot asserts the integer ``KEYSTONE_COUNT`` (= 4),
    AND the run is GROUNDED (visit.count > 0).

    A count of 7 ('they all look long'), 3 (one arm dropped) or 5 (Ōnaruto wrongly counted in)
    all fail. Grounding is required so an ungrounded parametric guess of '4' banks nothing.
    """
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    return KEYSTONE_COUNT in _int_values(_strip_list_markers(_primary_text(result)))


def _verdict_for(text: str, entity: Dict[str, Any]) -> str:
    """Return 'above' / 'below' / '' — the agent's stated pass/fail verdict for ``entity``.

    Two layouts are recognised, in priority order:

      1. NAME-then-verdict (per-row tables: "Ōnaruto Bridge — 876 m — NO"). A period-bounded,
         newline-TOLERANT window of up to 90 chars after the name is scanned and the EARLIEST
         cue wins, so 'does not exceed' beats the bare 'exceed' nested inside it and a
         following row's verdict cannot outrank this row's own.
      2. VERDICT-then-name (two named lists: "Exceed 1,200 m: A, B, C. Below: D, E"). Only
         consulted when layout 1 found nothing: the LAST cue in the current segment — the text
         since the previous period or blank line — is used, so a list header labels every item
         under it while a period or paragraph break stops the label from leaking further.
    """
    for m in re.finditer(entity["name_rx"], text, re.IGNORECASE):
        after = re.match(r"[^.]{0,90}", text[m.end():])
        hit = _VERDICT_RX.search(after.group(0) if after else "")
        if hit:
            return "below" if hit.group("below") else "above"
        segment = re.split(r"\.|\n\s*\n", text[:m.start()])[-1]
        back = list(_VERDICT_RX.finditer(segment))
        if back:
            return "below" if back[-1].group("below") else "above"
    return ""


# ── validation functions ─────────────────────────────────────────────────────────────────────────

def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a seven-way fan-out wants one page per bridge."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {
        "check":  "visit_count",
        "passed": n >= 6,
        "score":  min(1.0, n / len(ENTITIES)),
        "reason": f"{n} visit(s) (target >= {len(ENTITIES)}: one page per bridge; >= 6 to pass)",
    }


def validate_keystone_count(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the exact count of bridges with a longest span > 1,200 m."""
    passed = _keystone_ok(result, observability)
    passing_names = ", ".join(e["name"] for e in PASSING)
    return {
        "check":  "keystone_count",
        "passed": passed,
        "score":  1.0 if passed else 0.0,
        "reason": (
            f"Count {KEYSTONE_COUNT} (bridges with longest span > {THRESHOLD:,} m) present in "
            "the primary answer"
            if passed else
            f"Count {KEYSTONE_COUNT} absent or wrong in the primary answer. The "
            f"{len(PASSING)} bridges exceeding {THRESHOLD:,} m are: {passing_names}. "
            "Beware: a naive 'all seven' gives 7 (wrong); dropping one arm gives 3 (wrong); "
            "counting Ōnaruto (876 m) as a passer gives 5 (wrong)."
        ),
    }


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the seven (bridge, span) pairs were gathered.

    A bridge is credited only when BOTH its name AND its infobox span appear in the response.
    The seven spans (1650, 1545, 1408, 1385, 876, 850, 712) are all distinct, so no
    cross-crediting is possible. Deliberately NOT gated on the keystone — this is the axis
    that separates a structured fan-out from a linear agent even when the count is botched.

    Credit is CAPPED BY visit count (``min(hits, n_visits)``) so a 0-visit parametric answer
    that merely recites figures cannot bank breadth credit without ever browsing.
    """
    text = _all_text(result)
    hits = [
        e["name"] for e in ENTITIES
        if re.search(e["name_rx"], text, re.IGNORECASE) and re.search(e["span_rx"], text)
    ]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(hits), n_visits)
    n = len(ENTITIES)
    return {
        "check":  "coverage",
        "passed": credited == n,
        "score":  credited / n,
        "reason": (
            f"{credited}/{n} (bridge, span) pairs gathered "
            f"({', '.join(hits[:credited]) if credited else 'none'}; "
            f"{len(hits)} text-matched, {n_visits} visit(s))"
        ),
    }


def validate_classification(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated per-item diagnostic: how many of the seven bridges were correctly LABELLED
    above/below the 1,200 m threshold, independent of whether the final count is right.

    An agent that gathers six arms and labels all six correctly but miscounts still scores 6/7
    here, while a linear agent that only reached three arms scores 3/7 — this is the coverage
    axis expressed on the classification step rather than on raw value retrieval. Items with no
    detectable verdict, and items labelled the wrong way, both score zero. Capped by visit count
    for the same anti-parametric reason as ``validate_coverage``.
    """
    text = _all_text(result)
    correct: List[str] = []
    wrong: List[str] = []
    missing: List[str] = []
    for e in ENTITIES:
        verdict = _verdict_for(text, e)
        expected = "above" if e["passes"] else "below"
        if not verdict:
            missing.append(e["name"])
        elif verdict == expected:
            correct.append(e["name"])
        else:
            wrong.append(e["name"])
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(correct), n_visits)
    n = len(ENTITIES)
    return {
        "check":  "classification",
        "passed": credited == n,
        "score":  credited / n,
        "reason": (
            f"{credited}/{n} bridges correctly classified vs the {THRESHOLD:,} m threshold "
            f"({len(wrong)} mislabelled: {', '.join(wrong) or 'none'}; "
            f"{len(missing)} unlabelled: {', '.join(missing) or 'none'}; {n_visits} visit(s))"
        ),
    }


def validate_passing_bridges(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: all four passing bridge names present. Short-circuits to 0 without the
    keystone, so a wrong count cannot bank partial credit for naming some right bridges."""
    if not _keystone_ok(result, observability):
        return {
            "check":  "passing_bridges",
            "passed": False,
            "score":  0.0,
            "reason": "Keystone absent -> passing-bridge list not credited",
        }
    text = _all_text(result)
    hits = [e["name"] for e in PASSING if re.search(e["name_rx"], text, re.IGNORECASE)]
    n = len(PASSING)
    return {
        "check":  "passing_bridges",
        "passed": len(hits) == n,
        "score":  len(hits) / n,
        "reason": f"{len(hits)}/{n} passing bridge names identified ({', '.join(hits) or 'none'})",
    }


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites at least 5 of the 7 bridge pages. Short-circuits without keystone."""
    if not _keystone_ok(result, observability):
        return {
            "check":  "citation",
            "passed": False,
            "score":  0.0,
            "reason": "Keystone absent -> source URL credit withheld",
        }
    text = _all_text(result).lower()
    cited = sum(1 for e in ENTITIES if re.search(e["slug_rx"], text))
    n = len(ENTITIES)
    return {
        "check":  "citation",
        "passed": cited >= 5,
        "score":  cited / n,
        "reason": f"{cited}/{n} bridge pages cited (>= 5 to pass)",
    }


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_count,
        validate_coverage,
        validate_classification,
        validate_passing_bridges,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge.
    return None


# ── compiled plan ────────────────────────────────────────────────────────────────────────────────

def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    SEVEN INDEPENDENT parallel leaves (every ``depends_on`` is EMPTY): one per bridge, each
    reading that bridge's longest span from its own infobox. All threshold and counting logic
    lives in the aggregation, which restates each bridge → its span BEFORE comparing, so the
    cheap executor never has to hold seven values and a threshold rule at once.

    Encodes STRUCTURE only: it names the seven GIVEN bridges and the GIVEN threshold, and leaks
    no span figure, no per-bridge verdict and no count. Leaf ids key on the bridge (a given),
    never on the answer.
    """
    leaves: List[Dict[str, Any]] = []
    for e in ENTITIES:
        leaves.append({
            "id": f"{e['key']}_span",
            "instruction": (
                f"Open the English Wikipedia article for the {e['name']} ({e['country']}) and "
                "read, directly from the infobox, its LONGEST SPAN in metres — the 'Longest "
                "span' field (the main span between the towers). Report ONLY that single figure "
                "in metres and the exact source URL. Do NOT report the bridge's total length, "
                "height, or clearance below. Do not guess from memory."
            ),
            "expect": f"LONGEST SPAN of the {e['name']} in metres (a single number) -- source URL",
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        # Deterministic composition: the executor applies the threshold and counts in Python over
        # the seven gathered spans, rendering each check and both named lists itself (zero extra
        # LLM calls) — free-text counting is the confirmed failure mode here even when every span
        # is correct. Encodes the GIVEN threshold only. If a leaf fails to resolve, the composer
        # returns nothing and the free-text recipe below runs unchanged.
        "agg_mode": "computed",
        "composition": {
            "op": "count_threshold",
            "answer_noun": "bridge",
            "value_label": "longest span",
            "unit": "m",
            "comparator": ">",
            "threshold": THRESHOLD,
            "items": [{"leaf": f"{e['key']}_span", "label": e["name"], "type": "number"}
                      for e in ENTITIES],
        },
        "aggregation": (
            "You now have, for each of the seven bridges, its longest span in metres and a "
            "source URL. Before applying any threshold or counting, RESTATE each bridge's span "
            "explicitly in this format:\n"
            + "".join(f"  {e['name']} -> [span] m\n" for e in ENTITIES)
            + "(substituting the figure you retrieved for each '[span]' placeholder).\n\n"
            "Then, FOR EACH BRIDGE IN TURN, state whether its longest span is strictly greater "
            f"than {THRESHOLD:,} metres. Finally, COUNT the bridges whose span exceeds "
            f"{THRESHOLD:,} m — that integer (0–7) is the keystone answer. Report:\n"
            "  (a) the count (a single integer, 0–7) as the primary keystone answer,\n"
            f"  (b) which bridges exceed {THRESHOLD:,} m and which do not (two named lists),\n"
            "  (c) the full list of seven spans as you restated them above, and\n"
            "  (d) the source URL for each bridge."
        ),
    }
