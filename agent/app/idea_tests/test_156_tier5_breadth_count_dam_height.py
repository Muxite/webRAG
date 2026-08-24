"""
Test 156: Tier 5 (breadth) — 7 INDEPENDENT per-item lookups + COUNT-WITH-CONDITION.
Level: graph   Weight: long   Difficulty: 8/10

NO URLs are given. SEVEN named dams; for each, the agent must open that dam's own page and
read the STRUCTURAL HEIGHT in metres from the infobox 'Height' field (one page-read per dam),
then AGGREGATE across all seven to report HOW MANY have a height STRICTLY GREATER THAN 220
metres. The answer is a COUNT (integer 0-7) that is COMPUTED from seven fully independent
lookups and is published nowhere.

WHY IT DISCRIMINATES (the "many small independent lookups" sub-shape):
  * Every arm is a ONE-hop page-read on its own page. There is ZERO cross-item dependency, so
    a graph can dispatch all seven in a single parallel wave, while a linear ReAct agent must
    serialize seven gather-hops into a capped step budget and hold seven three-digit figures
    in one degrading scratchpad before applying the threshold.
  * Sibling of 072 in shape (count-with-condition) but a DIFFERENT domain (dam structural
    heights, not lake depths), a different item set, a different threshold, and an explicitly
    fully-parallel compiled plan (every leaf has an empty ``depends_on``).

Ground truth — every figure verified against the live English Wikipedia infobox 'Height'
field on 2026-08-22, read from the canonical article URL:

  dam                 article title        country       height    > 220 m?   margin
  ------------------------------------------------------------------------------------
  Nurek Dam           Nurek Dam            Tajikistan    300 m      YES       +80 m
  Grande Dixence Dam  Grande Dixence Dam   Switzerland   285 m      YES       +65 m
  Enguri Dam          Enguri Dam           Georgia       271.5 m    YES       +51.5 m
  Vajont Dam          Vajont Dam           Italy         262 m      YES       +42 m
  Katse Dam           Katse Dam            Lesotho       185 m      NO        -35 m
  Karakaya Dam        Karakaya Dam         Turkey        158 m      NO        -62 m
  Gordon Dam          Gordon Dam           Australia     140 m      NO        -80 m
  ------------------------------------------------------------------------------------
  KEYSTONE = count of dams with structural height > 220 m = 4

THRESHOLD MARGINS: the smallest margin in the set is 35 m (Katse, below) and 42 m (Vajont,
above), i.e. every item clears or misses 220 m by >= 35 m. No plausible single-figure misread
of a three-digit infobox value (+-10-15 m) can flip ANY item's individual pass/fail status,
so the count is robust to one noisy extraction. The pass/fail boundary also falls in a wide
EMPTY band: no dam in the set has a height between 185 m and 262 m.

ANTI-PARAMETRIC: the seven dams span seven countries on five continents and none is a
household name in the way Hoover/Aswan/Itaipu are (those were deliberately excluded because
their heights are commonly recalled). Enguri (271.5 m) and Karakaya (158 m) in particular
carry page-specific figures that are not simultaneously recallable, and the derived COUNT (4)
is not a published fact anywhere. A parametric guess has no reliable path to it.

KEYSTONE (hard 0/1)  = the count 4 in the primary answer slot, AND grounding (>=1 visit).
COVERAGE (UN-gated)  = how many of the seven (dam, height) pairs were gathered; the seven
                       heights (300, 285, 271.5, 262, 185, 158, 140) are all distinct.
CLASSIFICATION (un-gated) = how many of the seven items' individual pass/fail verdicts were
                       correctly reported -- the breadth axis that survives a botched count.
Secondary (GATED on keystone) = passing-dam names, source URLs.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# --- verified fixtures -------------------------------------------------------------------
# 'height' = structural height in metres from the Wikipedia infobox 'Height' field
# (live-verified 2026-08-22).  'passes' = (height > THRESHOLD).  Nothing below is leaked
# into the task statement or the compiled plan.

THRESHOLD: int = 220  # metres, strictly greater-than

ENTITIES: List[Dict[str, Any]] = [
    {
        "key":       "nurek",
        "name":      "Nurek Dam",
        "country":   "Tajikistan",
        "height":    300.0,
        "passes":    True,
        "name_rx":   r"nurek",
        # \b around the short numeric token; the ".0"/",0" tail is not accepted as a match.
        "value_rx":  r"(?<!\d)300(?!\s*\d)(?!\.\d)(?!,\d)",
        "slug_rx":   r"wiki/nurek_dam",
    },
    {
        "key":       "grande_dixence",
        "name":      "Grande Dixence Dam",
        "country":   "Switzerland",
        "height":    285.0,
        "passes":    True,
        "name_rx":   r"grande\s*[- ]?\s*dixence",
        "value_rx":  r"(?<!\d)285(?!\s*\d)(?!\.\d)(?!,\d)",
        "slug_rx":   r"wiki/grande_dixence_dam",
    },
    {
        "key":       "enguri",
        # Article title is "Enguri Dam"; "Inguri Dam" redirects there.
        "name":      "Enguri Dam",
        "country":   "Georgia",
        "height":    271.5,
        "passes":    True,
        "name_rx":   r"[ei]nguri",
        # Accept the exact 271.5 or a rounded 271/272 -- all three are correct readings of
        # the same infobox figure and all three sit far above the threshold.
        "value_rx":  r"(?<!\d)271(?:[.,]5)?(?!\d)|(?<!\d)272(?!\d)",
        "slug_rx":   r"wiki/[ei]nguri_dam",
    },
    {
        "key":       "vajont",
        "name":      "Vajont Dam",
        "country":   "Italy",
        "height":    262.0,
        "passes":    True,
        "name_rx":   r"vaj?[oi]?ont|vajont",
        # Infobox says 262 m; 261.6 m is the widely quoted precise figure for the same dam.
        "value_rx":  r"(?<!\d)262(?!\s*\d)(?!\.\d)(?!,\d)|(?<!\d)261(?:[.,]6)?(?!\d)",
        "slug_rx":   r"wiki/vajont_dam",
    },
    {
        "key":       "katse",
        "name":      "Katse Dam",
        "country":   "Lesotho",
        "height":    185.0,
        "passes":    False,
        "name_rx":   r"katse",
        "value_rx":  r"(?<!\d)185(?!\s*\d)(?!\.\d)(?!,\d)",
        "slug_rx":   r"wiki/katse_dam",
    },
    {
        "key":       "karakaya",
        "name":      "Karakaya Dam",
        "country":   "Turkey",
        "height":    158.0,
        "passes":    False,
        "name_rx":   r"karakaya",
        "value_rx":  r"(?<!\d)158(?!\s*\d)(?!\.\d)(?!,\d)",
        "slug_rx":   r"wiki/karakaya_dam",
    },
    {
        "key":       "gordon",
        "name":      "Gordon Dam",
        "country":   "Australia (Tasmania)",
        "height":    140.0,
        "passes":    False,
        "name_rx":   r"gordon",
        "value_rx":  r"(?<!\d)140(?!\s*\d)(?!\.\d)(?!,\d)",
        "slug_rx":   r"wiki/gordon_dam",
    },
]

PASSING: List[Dict[str, Any]] = [e for e in ENTITIES if e["passes"]]        # 4 dams
FAILING: List[Dict[str, Any]] = [e for e in ENTITIES if not e["passes"]]    # 3 dams
KEYSTONE_COUNT: int = len(PASSING)                                          # 4

# Numeric-token extractor: maximal digit runs with internal separators.
_NUM_TOKEN_RX = re.compile(r"\d[\d.,]*\d|\d")


# --- metadata ----------------------------------------------------------------------------

def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id":          "156",
        "test_name": (
            "Tier 5: Independent 7-way count-with-condition — dams taller than 220 m"
        ),
        "difficulty_level": "8/10",
        "category":         "Breadth Fan-out & Count-with-condition",
        "level":            "graph",
        "weight":           "long",
    }


# --- task statement ----------------------------------------------------------------------

def get_task_statement() -> str:
    listing = "\n".join(
        f"  {i}. {e['name']} ({e['country']})" for i, e in enumerate(ENTITIES, 1)
    )
    return (
        "You are given NO URLs — search to find the pages you need, then READ them (do not "
        "guess from memory). For EACH of the following seven dams, open that dam's own "
        "Wikipedia page and read ONE figure directly from the infobox: its HEIGHT in metres "
        "(the structural height of the dam itself — NOT the reservoir's surface elevation, "
        "NOT the length of the crest, NOT the height above the foundation of a different "
        "structure):\n"
        f"{listing}\n\n"
        "The seven lookups are completely independent of one another — nothing you learn "
        "about one dam is needed to look up any other.\n\n"
        f"Then COUNT how many of the seven dams have a height STRICTLY GREATER THAN "
        f"{THRESHOLD} metres, and report:\n"
        f"  (a) the count — how many of the seven dams are taller than {THRESHOLD} m "
        "(a single integer 0–7; this is the keystone answer and belongs in your primary "
        "answer),\n"
        f"  (b) which specific dams exceed {THRESHOLD} m and which do not (two named lists),\n"
        "  (c) each dam's height in metres as read from its infobox, and\n"
        "  (d) the exact source URL for every page you read."
    )


# --- deliverables / success criteria -----------------------------------------------------

def get_required_deliverables() -> List[str]:
    return [
        f"Count of dams with height > {THRESHOLD} m (the keystone integer, 0–7)",
        f"Which dams exceed {THRESHOLD} m and which do not (two named lists)",
        "Each dam's height in metres (seven individual infobox values)",
        "Source URL for each dam's page",
    ]


def get_success_criteria() -> List[str]:
    passing_names = ", ".join(e["name"] for e in PASSING)
    failing_names = ", ".join(e["name"] for e in FAILING)
    return [
        f"At least {len(ENTITIES) - 1} pages visited (target {len(ENTITIES)}, one per dam)",
        f"Correctly reports the count as {KEYSTONE_COUNT} "
        f"(the number of dams taller than {THRESHOLD} m)",
        f"Identifies the {len(PASSING)} dams above the threshold: {passing_names}",
        f"Identifies the {len(FAILING)} dams below the threshold: {failing_names}",
        "Reports all seven dams' heights from their infoboxes",
        "Cites a source URL for each dam",
    ]


# --- shared text helpers -----------------------------------------------------------------

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
    """Plain integers present in ``text`` (decimal tokens dropped, grouping commas stripped)."""
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
    A single leading digit-marker on an otherwise terse answer ('4. Four dams qualify') is far
    more likely the genuine short answer than list enumeration and must NOT be stripped."""
    if len(_LIST_MARKER_RX.findall(text)) < 2:
        return text
    return _LIST_MARKER_RX.sub("", text)


def _n_visits(observability: Dict[str, Any] = None) -> int:
    return int((observability or {}).get("visit", {}).get("count", 0) or 0)


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE gate: the primary answer slot contains the integer KEYSTONE_COUNT (= 4).

    Also requires GROUNDING (visit.count > 0): the integer alone is insufficient, else an
    ungrounded parametric guess of '4' would earn full keystone credit.
    """
    if _n_visits(observability) <= 0:
        return False
    return KEYSTONE_COUNT in _int_values(_strip_list_markers(_primary_text(result)))


# --- validation functions ----------------------------------------------------------------

def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a seven-way independent fan-out wants one page per dam."""
    n = _n_visits(observability)
    need = len(ENTITIES) - 1  # >= 6
    return {
        "check":  "visit_count",
        "passed": n >= need,
        "score":  min(1.0, n / len(ENTITIES)),
        "reason": (
            f"{n} visit(s) (target >= {len(ENTITIES)}: one page per dam; >= {need} to pass)"
        ),
    }


def validate_keystone_count(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the exact count of dams taller than 220 m in the primary answer.

    Correct answer = 4. Off-by-one in either direction (3 / 5), the naive 'all seven' (7),
    and 0 all fail. Only a run that opens the pages, reads each height, applies the threshold
    per item and sums the booleans passes.
    """
    passed = _keystone_ok(result, observability)
    passing_names = ", ".join(e["name"] for e in PASSING)
    return {
        "check":  "keystone_count",
        "passed": passed,
        "score":  1.0 if passed else 0.0,
        "reason": (
            f"Count {KEYSTONE_COUNT} (dams taller than {THRESHOLD} m) present in the primary "
            "answer, with grounding"
            if passed else
            f"Count {KEYSTONE_COUNT} absent/wrong in the primary answer (or no page visited). "
            f"The {len(PASSING)} dams above {THRESHOLD} m are: {passing_names}."
        ),
    }


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the seven (dam, height) pairs were gathered.

    A dam is credited only when BOTH its name AND its infobox height appear in the response.
    The seven heights (300, 285, 271.5, 262, 185, 158, 140) are all distinct, so there is no
    cross-crediting. Deliberately NOT short-circuited on the keystone — this is the axis that
    separates a structured agent (which fanned out to all seven pages) from a linear one even
    when the final count is botched.

    Credit is CAPPED BY visit count so a 0-visit recall answer cannot bank breadth credit.
    """
    text = _all_text(result)
    hits = [
        e["name"] for e in ENTITIES
        if re.search(e["name_rx"], text, re.IGNORECASE) and re.search(e["value_rx"], text)
    ]
    credited = min(len(hits), _n_visits(observability))
    n = len(ENTITIES)
    return {
        "check":  "coverage",
        "passed": credited == n,
        "score":  credited / n,
        "reason": (
            f"{credited}/{n} (dam, height) pairs gathered "
            f"({', '.join(hits[:credited]) if credited else 'none'}; "
            f"{len(hits)} text-matched, {_n_visits(observability)} visit(s))"
        ),
    }


# Per-item verdict cues. Only TRUE classification triggers are listed: a bare
# "Nurek Dam -> 300 m" coverage row must NOT be read as a verdict, which is why the bare
# comparison glyphs '>' and '<' are deliberately EXCLUDED (the '->' row separator would
# otherwise forge a "yes" verdict for every dam, including the failing ones).
_YES_CUE = (
    r"yes\b|✓|✔|\btrue\b|above\b|exceeds?\b|taller\b|\bover\b|greater\b|pass(?:es|ed)?\b"
)
_NO_CUE = (
    r"\bno\b|✗|✘|\bfalse\b|below\b|under\b|shorter\b|beneath\b"
    r"|does\s+not\s+exceed|not\s+taller|not\s+greater|fail(?:s|ed)?\b"
)
_YES_RX = re.compile(_YES_CUE, re.IGNORECASE)
_NO_RX = re.compile(_NO_CUE, re.IGNORECASE)

# Proximity window (chars) searched on each side of a dam name for its verdict cue when the
# name's own line carries no cue. Newlines are tolerated on purpose ('Above 220 m:\n  Nurek
# Dam' is a common report layout); the window is clipped at the nearest SENTENCE-ending period
# so a verdict about one dam cannot bleed into the next dam's sentence.
_VERDICT_WINDOW = 160
# A sentence-ending period is one NOT followed by a digit, so decimal figures like "271.5"
# (and thus the Enguri row) are never mistaken for a sentence boundary.
_SENT_END_RX = re.compile(r"\.(?!\d)")


def _direction_in(text: str, span, lo: int, hi: int):
    """Direction of the verdict cue NEAREST to ``span`` within ``text[lo:hi]``, or None when
    that region holds no cue. Distance is measured in characters from the dam name."""
    s, e = span
    pre, post = text[lo:s], text[e:hi]
    best_dist, best_yes = None, None
    for rx, is_yes in ((_YES_RX, True), (_NO_RX, False)):
        dists = [len(pre) - m.end() for m in rx.finditer(pre)]
        dists += [m.start() for m in rx.finditer(post)]
        if not dists:
            continue
        d = min(dists)
        if best_dist is None or d < best_dist:
            best_dist, best_yes = d, is_yes
    return best_yes


def _nearest_cue_is_yes(text: str, span):
    """Verdict direction attached to one dam-name occurrence, or None if none is stated.

    The dam's OWN LINE wins first: in a two-list layout ('Above 220 m: A, B, C, D' /
    'Below 220 m: E, F, G') the last name on the first line sits closer to the *next* line's
    heading than to its own, so a pure character-distance rule would flip it. Only when the
    name's line carries no cue at all does the search widen to a sentence-clipped window,
    which covers the block layout where the heading sits on its own line above the names.
    """
    s, e = span
    line_lo = text.rfind("\n", 0, s) + 1
    nl = text.find("\n", e)
    line_hi = len(text) if nl == -1 else nl
    same_line = _direction_in(text, span, line_lo, line_hi)
    if same_line is not None:
        return same_line
    lo = max((m.end() for m in _SENT_END_RX.finditer(text, 0, s)), default=0)
    lo = max(lo, s - _VERDICT_WINDOW)
    m = _SENT_END_RX.search(text, e)
    hi = min(m.start() if m else len(text), e + _VERDICT_WINDOW)
    return _direction_in(text, span, lo, hi)


def _verdict_ok(text: str, entity: Dict[str, Any]) -> bool:
    """True when the response states this dam's own threshold verdict in the correct direction.

    The FIRST occurrence of the dam name that carries any verdict cue decides — a later,
    contradicting restatement cannot rescue a wrong verdict, and a correct verdict cannot be
    forged by a stray cue elsewhere in a long report.
    """
    for m in re.finditer(entity["name_rx"], text, re.IGNORECASE):
        verdict = _nearest_cue_is_yes(text, m.span())
        if verdict is None:
            continue
        return verdict is entity["passes"]
    return False


def validate_item_classification(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated per-item diagnostic: how many of the seven dams' individual above/below-threshold
    verdicts were reported CORRECTLY, independent of whether the final count is right.

    This is the second breadth axis: a linear agent that ran out of budget reports two or three
    verdicts; a structured agent that fanned out reports all seven even when it then miscounts.
    Capped by visit count for the same anti-recall reason as ``validate_coverage``.
    """
    text = _all_text(result)
    hits = [e["name"] for e in ENTITIES if _verdict_ok(text, e)]
    credited = min(len(hits), _n_visits(observability))
    n = len(ENTITIES)
    return {
        "check":  "item_classification",
        "passed": credited == n,
        "score":  credited / n,
        "reason": (
            f"{credited}/{n} per-dam threshold verdicts stated correctly "
            f"({', '.join(hits[:credited]) if credited else 'none'})"
        ),
    }


def validate_passing_dams(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: all four above-threshold dam names present. Short-circuits to 0 when
    the keystone is absent, so a wrong count cannot bank partial credit here."""
    if not _keystone_ok(result, observability):
        return {
            "check":  "passing_dams",
            "passed": False,
            "score":  0.0,
            "reason": "Keystone absent -> above-threshold dam list not credited",
        }
    text = _all_text(result)
    hits = [e["name"] for e in PASSING if re.search(e["name_rx"], text, re.IGNORECASE)]
    n = len(PASSING)
    return {
        "check":  "passing_dams",
        "passed": len(hits) == n,
        "score":  len(hits) / n,
        "reason": f"{len(hits)}/{n} above-threshold dams named ({', '.join(hits) or 'none'})",
    }


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: source URLs. Short-circuits to 0 when the keystone is absent."""
    if not _keystone_ok(result, observability):
        return {
            "check":  "citations",
            "passed": False,
            "score":  0.0,
            "reason": "Keystone absent -> source URL credit withheld",
        }
    text = _all_text(result).lower()
    cited = sum(1 for e in ENTITIES if re.search(e["slug_rx"], text))
    n = len(ENTITIES)
    return {
        "check":  "citations",
        "passed": cited >= 5,
        "score":  cited / n,
        "reason": f"{cited}/{n} dam pages cited (>= 5 to pass)",
    }


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_count,
        validate_coverage,
        validate_item_classification,
        validate_passing_dams,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge.
    return None


# --- compiled plan -----------------------------------------------------------------------

def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    SEVEN leaves, EVERY ONE with an empty ``depends_on``: the arms are genuinely independent,
    so the whole plan is a single parallel wave followed by the aggregation. Each leaf ``id``
    is keyed on the DAM (a GIVEN), never on a height or the count. All comparison and counting
    logic lives ONLY in the aggregation step, which is forced to restate each dam -> its height
    before applying the threshold.

    Encodes STRUCTURE and the GIVEN threshold only: no height value, no per-dam verdict and no
    count are leaked anywhere in the plan text.
    """
    leaves = [
        {
            "id": f"{e['key']}_height",
            "instruction": (
                f"Open the Wikipedia article for {e['name']} ({e['country']}) and read, "
                "directly from the infobox, the dam's HEIGHT in metres — the 'Height' field "
                "(the structural height of the dam itself). Report ONLY that single height "
                "figure in metres and the exact source URL. Do NOT report the reservoir's "
                "surface elevation, the crest length, or any other dimension. Do not guess "
                "from memory."
            ),
            "expect": f"HEIGHT of {e['name']} in metres (a single number) -- source URL",
            "depends_on": [],
        }
        for e in ENTITIES
    ]
    restate = "\n".join(f"  {e['name']} -> [height] m" for e in ENTITIES)
    return {
        "leaves": leaves,
        # Deterministic composition: the executor applies the threshold and counts in Python
        # over the seven gathered heights (zero extra LLM calls) — free-text counting is the
        # confirmed failure mode here even when every height is correct. If any leaf fails to
        # resolve, the composer returns nothing and the recipe below runs unchanged.
        "agg_mode": "computed",
        "composition": {
            "op": "count_threshold",
            "answer_noun": "dam",
            "value_label": "height",
            "unit": "m",
            "comparator": ">",
            "threshold": THRESHOLD,
            "items": [
                {"leaf": f"{e['key']}_height", "label": e["name"], "type": "number"}
                for e in ENTITIES
            ],
        },
        "aggregation": (
            "You now have, for each of the seven dams, its height in metres and a source URL. "
            "Before applying any threshold or counting, RESTATE each dam's height explicitly "
            "in this format:\n"
            f"{restate}\n"
            "(substituting the figure you retrieved for each '[height]' placeholder).\n\n"
            "Then, FOR EACH DAM IN TURN, state whether its height is strictly greater than "
            f"{THRESHOLD} metres. Finally, COUNT the dams whose height exceeds {THRESHOLD} m "
            "— that integer (0–7) is the keystone answer. Report:\n"
            "  (a) the count (a single integer, 0–7) as the primary keystone answer,\n"
            f"  (b) which dams exceed {THRESHOLD} m and which do not (two named lists),\n"
            "  (c) the full list of seven heights as you restated them above, and\n"
            "  (d) the source URL for each dam."
        ),
    }
