r"""
Test 148: Tier 5 (graph) — SURVIVOR -> CONFLICTING-SOURCE RECONCILIATION -> CONSTRAINED SUBSET-SUM.
Level: graph   Weight: long   Difficulty: 10/10

COMPOUND SHAPE — THREE different axis TYPES stacked in one 8-visit budget. The suite today has each
axis alone: branch-eliminate survivors (095, 122-127), conflicting-source reconciliation (128-133)
and bounded subset-sums (070, 083). None combines them, and each of the three is load-bearing here
— skip any one and the keystone number is wrong by >1,500 km:

    STAGE 1  SURVIVOR (categorical elimination) — four great rivers; exactly ONE empties into the
             KARA SEA. Page-only: it lives in each river's infobox 'mouth' row.
    STAGE 2  CONFLICTING-SOURCE RECONCILIATION — two lengths circulate for the survivor: the length
             of the RIVER ITSELF and the much larger length of the combined RIVER SYSTEM measured
             through its longest tributary. Both are printed on the survivor's own page. The rule:
             use the river itself. Getting this wrong shifts the keystone by +1,710 km.
    STAGE 3  CONSTRAINED SUBSET-SUM — four further named rivers; include one ONLY IF it satisfies
             BOTH (i) it lies in the SURVIVOR's drainage basin (its mouth must be checked, possibly
             transitively) AND (ii) its own length exceeds 2,000 km. Add the survivor's reconciled
             length. That total is the keystone.

  Golden path = 8 visits (4 candidate rivers + 4 component rivers; the survivor's page is read in
  stage 1 and re-used in stage 2), inside the suite's 6-9 visit budget.

Ground truth (verified against live English Wikipedia via the MediaWiki API, 2026-08-06 — infobox
'mouth' and 'length' rows, plus the Ob article's own prose for the system-length figure):

  STAGE 1 — candidate rivers, infobox mouth:
    Ob      -> Gulf of Ob  (the article's discharge row reads "Ob Estuary, Gulf of Ob (KARA SEA)";
                            the Gulf of Ob's own infobox lists its ocean as the Kara Sea)  SURVIVES
    Lena    -> Lena Delta, which merges into the LAPTEV SEA                                eliminated
    Amur    -> STRAIT OF TARTARY (Pacific, Sea of Okhotsk basin)                           eliminated
    Kolyma  -> EAST SIBERIAN SEA                                                           eliminated
  Categorical, not numeric: three mouths are explicit non-matches, so a single noisy read cannot
  flip the survivor.

  STAGE 2 — the survivor's two circulating lengths (both on the Ob's own article):
    infobox 'length'                     = 3,700 km   <- the RIVER ITSELF (the correct rule)
    lede + body: "with its tributary the Irtysh forms the world's seventh-longest river SYSTEM, at
    5,410 km" / "The combined Ob-Irtysh system ... is 5,410 km long"   <- the SYSTEM (the decoy)
  Some sources quote 3,650 km for the river itself; both variants land inside the keystone band
  (see KEYSTONE_BAND), while 5,410 is 1,710 km away and cannot.

  STAGE 3 — the four component rivers (each with its own article):
    river    length      mouth (basin test)                  in Ob basin?  > 2,000 km?  counted?
    Irtysh   4,248 km    -> Ob                               YES           YES          YES
    Ishim    2,450 km    -> Irtysh (-> Ob)                   YES           YES          YES
    Tobol    1,591 km    -> Irtysh (-> Ob)                   YES           no           no
    Vilyuy   2,650 km    -> Lena                             no            YES          no

  KEYSTONE = 3,700 + 4,248 + 2,450 = 10,398 km.

ROBUSTNESS / margins — the accepted band is [10,330, 10,410] (the exact total, the 3,650-variant
total 10,348, and +/-10 km of rounding slack). EVERY single-error total lands far outside it:
    used the SYSTEM length (5,410)               -> 12,108   (+1,698 past the band edge)
    included the Tobol (length rule dropped)     -> 11,989   (+1,579)
    included the Vilyuy (basin rule dropped)     -> 13,048   (+2,638)
    both membership rules dropped                -> 14,639   (+4,229)
    dropped the Ishim                            ->  7,948   (-2,382)
    dropped the Irtysh                           ->  6,150   (-4,180)
    survivor's length alone / system alone       ->  3,700 / 5,410
  The nearest wrong total is 1,579 km from the band, i.e. ~15% of the keystone — no plausible
  extraction slip lands in band, and no wrong subset coincides with the right total. The four
  component lengths (1,591 / 2,450 / 2,650 / 4,248) are pairwise distinct, so the coverage
  diagnostic is collision-free.

WHY IT DISCRIMINATES:
  * The famous, quotable figure in this domain is the SYSTEM length ("seventh-longest river system
    in the world, 5,410 km") — exactly the decoy; the river's own 3,700 km is the boring one.
  * A one-round agent that elects a survivor and stops reports 3,700 km (or 5,410) and fails.
  * A subset-sum agent that skips the basin test banks the Vilyuy (a Lena tributary) purely because
    it is long; one that skips the length test banks the Tobol. Both are detectable.
  * Nobody memorises "10,398 km": the keystone is a computed quantity over four page-only figures,
    so it is parametric-leak resistant by construction.

HONEST NOTE ON THE STAGE-1 LEAK SURFACE: three of the four component rivers named in stage 3 are in
fact Ob-basin rivers, so a model with strong parametric geography could shortcut the stage-1
elimination. That shortcut buys nothing on its own — the keystone still requires four page-only
lengths, the length/basin membership tests and the stage-2 reconciliation — and the un-gated
mouth-breadth diagnostic separately measures whether all four candidate mouths were actually read.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


LENGTH_THRESHOLD = 2000   # km — a component counts only if its own length EXCEEDS this

# ── STAGE 1: candidate rivers (the elimination set) ──
CANDIDATES: List[Dict[str, Any]] = [
    {
        "key": "ob", "name": "Ob", "desc": "the Ob, in western Siberia, Russia",
        "name_rx": r"\bob\b", "mouth_rx": r"kara\s+sea|gulf\s+of\s+ob|ob\s+estuary",
        "slug_rx": r"wiki/ob_\(river\)|wiki/ob_river", "survivor": True,
    },
    {
        "key": "lena", "name": "Lena", "desc": "the Lena, in eastern Siberia, Russia",
        "name_rx": r"\blena\b", "mouth_rx": r"laptev|lena\s+delta",
        "slug_rx": r"wiki/lena_\(river\)|wiki/lena_river", "survivor": False,
    },
    {
        "key": "amur", "name": "Amur", "desc": "the Amur, on the Russia/China border",
        "name_rx": r"\bamur\b", "mouth_rx": r"tartary|okhotsk|nikolayevsk|pacific",
        "slug_rx": r"wiki/amur", "survivor": False,
    },
    {
        "key": "kolyma", "name": "Kolyma", "desc": "the Kolyma, in far north-eastern Siberia, Russia",
        "name_rx": r"\bkolyma\b", "mouth_rx": r"east\s+siberian",
        "slug_rx": r"wiki/kolyma", "survivor": False,
    },
]
SURVIVOR = next(c for c in CANDIDATES if c["survivor"])      # the Ob

# ── STAGE 3: the component rivers (the subset-sum candidate set) ──
# ``counted`` is DERIVED below from the two membership rules, never hand-set.
COMPONENTS: List[Dict[str, Any]] = [
    {"key": "irtysh", "name": "Irtysh", "length": 4248, "in_basin": True,
     "name_rx": r"irtysh", "len_rx": r"\b4[,.\s]?248\b", "flows_into": "the Ob",
     "slug_rx": r"wiki/irtysh"},
    {"key": "ishim", "name": "Ishim", "length": 2450, "in_basin": True,
     "name_rx": r"ishim", "len_rx": r"\b2[,.\s]?450\b", "flows_into": "the Irtysh (and so the Ob)",
     "slug_rx": r"wiki/ishim"},
    {"key": "tobol", "name": "Tobol", "length": 1591, "in_basin": True,
     "name_rx": r"tobol\b", "len_rx": r"\b1[,.\s]?591\b", "flows_into": "the Irtysh (and so the Ob)",
     "slug_rx": r"wiki/tobol"},
    {"key": "vilyuy", "name": "Vilyuy", "length": 2650, "in_basin": False,
     "name_rx": r"vil[iy]u", "len_rx": r"\b2[,.\s]?650\b", "flows_into": "the Lena",
     "slug_rx": r"wiki/vilyuy"},
]

SURVIVOR_LENGTH = 3700          # km — the RIVER ITSELF (infobox), the reconciled value
SURVIVOR_SYSTEM_LENGTH = 5410   # km — the combined river SYSTEM (the conflicting decoy)

_COUNTED = [c for c in COMPONENTS if c["in_basin"] and c["length"] > LENGTH_THRESHOLD]
KEYSTONE_TOTAL = SURVIVOR_LENGTH + sum(c["length"] for c in _COUNTED)   # 10,398 km
assert KEYSTONE_TOTAL == 10398, f"fixture drift: keystone total is {KEYSTONE_TOTAL}"
# Band: the exact total, the 3,650-variant total (10,348) and +/-10 km of rounding slack. The
# nearest wrong total (11,989 — the Tobol wrongly included) is 1,579 km outside it.
KEYSTONE_BAND = set(range(10330, 10411))

# Numeric-token extractor (mirrors 070): maximal digit runs with internal separators, bounded by
# digits, so a trailing sentence period is not swallowed. ``_int_values`` keeps only plain integers.
_NUM_TOKEN_RX = re.compile(r"\d[\d.,]*\d|\d")
_LIST_MARKER_RX = re.compile(r"(?m)^\s*\(?(\d{1,2})[.)]\s+")

# Reconciliation tokens.
_RIVER_ITSELF_RX = re.compile(r"\b3[,.\s]?700\b|\b3[,.\s]?650\b")
_SYSTEM_RX = re.compile(r"\b5[,.\s]?410\b")
# The system figure must be FLAGGED as the combined/system/excluded one — a rule marker within a
# sentence of the token, in either direction ([^.] is newline-tolerant, sentence-bounded).
_RULE_MARK = (r"(?:system|combined|together|includ\w+|tributary|not\s+the|exclude\w*|excluded|"
              r"reject\w*|wrong|decoy|seventh|world'?s|joint|whole)")
_SYSTEM_TOK = r"(?:5[,.\s]?410)"
_SYSTEM_FLAGGED_RX = re.compile(
    rf"{_RULE_MARK}[^.]{{0,90}}{_SYSTEM_TOK}|{_SYSTEM_TOK}[^.]{{0,90}}{_RULE_MARK}",
    re.IGNORECASE,
)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "148",
        "test_name": "Tier 5: Survivor -> conflicting-source reconciliation -> constrained subset-sum (Kara Sea river basin)",
        "difficulty_level": "10/10",
        "category": "Compound: categorical survivor + source reconciliation + constrained subset-sum",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    cand = "\n".join(f"  {i}. the {c['name']} — {c['desc']}" for i, c in enumerate(CANDIDATES, 1))
    comp = "\n".join(f"  - the {c['name']}" for c in COMPONENTS)
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess from "
        "memory). Three stages; each stage depends on the one before it.\n\n"
        "STAGE 1 — eliminate to one survivor. Four great rivers:\n"
        f"{cand}\n"
        "Exactly ONE of these four empties into the KARA SEA. Open EACH river's page and read its "
        "MOUTH from the infobox to decide which one — the other three empty into the Laptev Sea, "
        "the East Siberian Sea, or the Strait of Tartary (the Pacific side). Determine the mouth of "
        "all four; do not simply pick the one you have heard of.\n\n"
        "STAGE 2 — reconcile two conflicting lengths for the surviving river. Two different lengths "
        "circulate for it and BOTH appear on its own article: one is the length of the RIVER "
        "ITSELF (source confluence to mouth); the other, much larger, is the length of the combined "
        "RIVER SYSTEM measured through its longest tributary — a figure widely quoted because that "
        "system ranks among the longest in the world. Apply this rule: use the length of the RIVER "
        "ITSELF; the combined-system figure is NOT the river's length. Do not average them.\n\n"
        "STAGE 3 — compute the keystone. Consider these four further rivers:\n"
        f"{comp}\n"
        "Open EACH one's page and read (i) its own LENGTH in km and (ii) the river or sea it flows "
        "into. Count a river ONLY IF it satisfies BOTH conditions:\n"
        "  (A) it lies within the SURVIVING river's drainage basin — i.e. its water ultimately "
        "reaches the surviving river (this can be indirect: a river that flows into a tributary of "
        "the survivor is still in the basin), AND\n"
        f"  (B) its own length is MORE THAN {LENGTH_THRESHOLD:,} km.\n"
        "Then ADD UP: the surviving river's own reconciled length (from stage 2) PLUS the lengths of "
        "every counted river. Not all four qualify — some fail (A), some fail (B).\n\n"
        "Report (a) that total in km (a single number — the keystone answer); (b) which river "
        "survived stage 1 and the mouth of each of the four candidates; (c) the two conflicting "
        "lengths from stage 2, saying which one you used and why the other was rejected; and (d) for "
        "each of the four stage-3 rivers, its length, what it flows into, and whether you counted "
        "it; citing the exact Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The computed total in km (survivor's own length + the qualifying rivers' lengths) — the keystone",
        "Which river empties into the Kara Sea (the survivor) + the mouth of each of the four candidates",
        "The two conflicting lengths for the survivor, with the river-itself figure chosen and the "
        "combined-system figure identified and rejected",
        "For each of the four stage-3 rivers: its length, what it flows into, and the include/exclude verdict",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 5 pages visited (golden path 8: four candidate rivers + four component rivers)",
        "Determines the mouth of ALL FOUR candidates and elects the Kara Sea one (the Ob)",
        "Uses the river's own length (3,700 km), NOT the combined river-system length (5,410 km)",
        "Applies BOTH membership rules: excludes the Tobol (1,591 km, too short) and the Vilyuy "
        "(2,650 km but in the Lena basin); includes the Irtysh (4,248 km) and the Ishim (2,450 km)",
        "Reports the total 10,398 km (band [10,330, 10,410])",
        "Cites the survivor's page and the component pages",
    ]


def _primary_text(result: Dict[str, Any]) -> str:
    """Primary answer text — prefer ``deliverables[0]`` (the keystone slot), else final_deliverable."""
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list) and deliv and deliv[0] is not None:
            return str(deliv[0])
    return extract_final_text(result)


def _all_text(result: Dict[str, Any]) -> str:
    """Full reported text — final deliverable plus every deliverable slot."""
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return " ".join(parts)


def _int_values(text: str) -> List[int]:
    """Plain integer values in ``text``: decimals dropped, grouping separators stripped."""
    vals: List[int] = []
    for tok in _NUM_TOKEN_RX.findall(text):
        core = tok.replace(",", "").replace(" ", "")
        if "." in core:
            continue                    # a decimal like "10398.5" is not a plain integer token
        try:
            vals.append(int(core))
        except ValueError:
            continue
    return vals


def _strip_list_markers(text: str) -> str:
    """Drop leading enumeration markers ('1. ', '(2) ') only when the text really looks like a
    numbered list (>=2 markers) — a single leading digit-marker on a terse answer is far more
    likely the answer itself (the 070 regression)."""
    if len(_LIST_MARKER_RX.findall(text)) < 2:
        return text
    return _LIST_MARKER_RX.sub("", text)


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE gate (hard 0/1): the primary answer carries a total inside [10,330, 10,410] AND the
    run is GROUNDED (visit.count > 0). Every wrong survivor, wrong length reconciliation or wrong
    subset membership lands >= 1,579 km outside the band."""
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    return any(v in KEYSTONE_BAND for v in _int_values(_strip_list_markers(_primary_text(result))))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: the golden path reads eight pages (four candidates + four components)."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 5, "score": min(1.0, n / 8.0),
            "reason": f"{n} visit(s) (target 8: four candidate rivers + four component rivers; >=5 to pass)"}


def validate_keystone_total(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the computed total 10,398 km — the survivor's reconciled own length plus
    the lengths of the basin rivers over 2,000 km. Rejects the system-length total (12,108), the
    Tobol-included total (11,989), the Vilyuy-included total (13,048) and every drop-one total."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_total", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": ("Total 10,398 km present (band [10,330, 10,410])" if passed
                       else "Keystone total (10,398 km) missing/incorrect — beware the combined-system "
                            "length (-> 12,108), the too-short Tobol (-> 11,989) and the out-of-basin "
                            "Vilyuy (-> 13,048)")}


def validate_branch_exploration(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic #1 (the elimination axis): how many of the FOUR candidate rivers
    were resolved to their mouth (name AND mouth token present). Retained even when the arithmetic
    downstream is botched — it is the axis that separates a structured agent that checks all four
    candidates from a linear one that guesses the famous river. Visit-capped (``min(hits,
    n_visits)``) because a mouth is a page-only infobox fact."""
    text = _all_text(result)
    hits = [c["name"] for c in CANDIDATES
            if re.search(c["name_rx"], text, re.IGNORECASE)
            and re.search(c["mouth_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(hits), n_visits)
    n = len(CANDIDATES)
    return {"check": "branch_exploration", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} candidate rivers resolved to their mouth from visited pages "
                      f"({', '.join(hits[:credited]) or 'none'}; {len(hits)} text-matched, "
                      f"{n_visits} visit(s))"}


def validate_component_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic #2 (the computation axis): how many of the FOUR component rivers
    had their own length gathered (name AND its distinct length figure). The four lengths (1,591 /
    2,450 / 2,650 / 4,248) are pairwise distinct, so attribution is collision-free. NOT gated on the
    keystone — it credits the gathering even when the membership rules or the addition are botched.
    Visit-capped for the same reason as above."""
    text = _all_text(result)
    hits = [c["name"] for c in COMPONENTS
            if re.search(c["name_rx"], text, re.IGNORECASE)
            and re.search(c["len_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(hits), n_visits)
    n = len(COMPONENTS)
    return {"check": "component_coverage", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} component rivers' lengths gathered from visited pages "
                      f"({', '.join(hits[:credited]) or 'none'}; {len(hits)} text-matched, "
                      f"{n_visits} visit(s))"}


def validate_reconciliation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary (the reconciliation axis): the answer surfaces the survivor's own length
    (3,700 km / 3,650 km) AND flags the combined-system figure (5,410 km) as the rejected one — a
    rule marker within a sentence of that token. Short-circuits to 0 without the keystone, so a run
    that mis-computes the total cannot bank reconciliation credit."""
    if not _keystone_ok(result, observability):
        return {"check": "reconciliation", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> length reconciliation not credited"}
    text = _all_text(result)
    has_river = bool(_RIVER_ITSELF_RX.search(text))
    flagged = bool(_SYSTEM_FLAGGED_RX.search(text))
    hits = int(has_river) + int(flagged)
    return {"check": "reconciliation", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"river-itself length reported={has_river}, combined-system figure flagged={flagged}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the pages the three stages had to read (four candidates + four
    components). Short-circuits to 0 without the keystone."""
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    slugs = [c["slug_rx"] for c in CANDIDATES] + [c["slug_rx"] for c in COMPONENTS]
    cited = sum(1 for s in slugs if re.search(s, text))
    return {"check": "citations", "passed": cited >= 3, "score": min(1.0, cited / 5.0),
            "reason": f"{cited}/8 source page(s) cited (need >=3; target >=5)"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_total,
        validate_branch_exploration,
        validate_component_coverage,
        validate_reconciliation,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored SURVIVOR + RECONCILE + SUM DAG scaffold for the ``graph_compiled`` variant.

    Two waves — a wide gathering wave of EIGHT independent leaves, then ONE dependent leaf that can
    only run after the elimination resolves (so this is neither a pure fan-out nor a pure chain):
      * WAVE 1a — four leaves, one per GIVEN candidate river, each reading ONLY that river's mouth.
      * WAVE 1b — four leaves, one per GIVEN component river, each reading BOTH that river's own
        length AND the river/sea it flows into (the two facts its membership test needs), and
        restating its own name so aggregation cannot re-bind a stray number.
      * WAVE 2 — ONE dependent leaf that templates the four mouths, elects the Kara Sea survivor and
        reads, from THAT river's page, BOTH circulating lengths (the river itself and the combined
        system) — the reconciliation input. It never states which figure to use downstream beyond
        the GIVEN rule.
      * ALL arithmetic and both membership rules live only in the aggregation, which must write out
        each component's two-way check and the addition explicitly before concluding.

    Leak-free: names the GIVEN candidates, the GIVEN component rivers, the GIVEN Kara Sea criterion
    and the GIVEN 2,000 km threshold, but no mouth, no length, not which river survives, not which
    components qualify, and not the keystone total."""
    mouth_leaves = [
        {
            "id": f"mouth_{c['key']}",
            "instruction": (
                f"Open the English Wikipedia page for the {c['name']} river — {c['desc']}. Read its "
                "MOUTH (the body of water it ultimately empties into) directly from the infobox, "
                "and note which sea that mouth belongs to if the infobox names a gulf, estuary or "
                f"delta. Report in the form 'the {c['name']}: mouth = <BODY OF WATER> (sea: <SEA>)' "
                "followed by the exact source URL. Report no other fact and do not guess from memory."
            ),
            "expect": f"'the {c['name']}: mouth = <BODY OF WATER> (sea: <SEA>)' — source URL",
            "depends_on": [],
        }
        for c in CANDIDATES
    ]
    component_leaves = [
        {
            "id": f"comp_{c['key']}",
            "instruction": (
                f"Open the English Wikipedia page for the {c['name']} river. Read TWO facts from the "
                "infobox: (A) its own LENGTH in kilometres, and (B) the river or sea it FLOWS INTO "
                f"(its mouth). Report in the exact form 'the {c['name']}: length = <number> km; "
                "flows into = <RIVER OR SEA>' followed by the exact source URL, so the river's name "
                "stays attached to both figures. Do not report the length of any other river and do "
                "not guess from memory."
            ),
            "expect": f"'the {c['name']}: length = <number> km; flows into = <RIVER OR SEA>' — source URL",
            "depends_on": [],
        }
        for c in COMPONENTS
    ]
    survivor_leaf = {
        "id": "survivor_lengths",
        "instruction": (
            "You are given the four candidate rivers and the mouth of each:\n"
            + "\n".join(f"  the {c['name']} -> {{mouth_{c['key']}}}" for c in CANDIDATES)
            + "\nDetermine which SINGLE one empties into the KARA SEA. Open THAT surviving river's "
              "Wikipedia page and read BOTH lengths that its article gives for it: (A) the length "
              "of the RIVER ITSELF, from the infobox, and (B) the larger length quoted for the "
              "combined RIVER SYSTEM measured through its longest tributary, which appears in the "
              "article text. Report the surviving river's name, figure (A) labelled 'river itself', "
              "figure (B) labelled 'combined system', and the exact source URL. Do not average them "
              "and do not guess from memory."
        ),
        "expect": "SURVIVING river (the Kara Sea one) + 'river itself = <km>' + 'combined system = <km>' — source URL",
        "depends_on": [f"mouth_{c['key']}" for c in CANDIDATES],
    }
    return {
        "leaves": mouth_leaves + component_leaves + [survivor_leaf],
        "aggregation": (
            "You now have (1) each candidate river's mouth, (2) which one empties into the Kara Sea "
            "(the survivor) together with BOTH lengths quoted for it, and (3) for each of the four "
            "further rivers, its own length and what it flows into.\n"
            "STEP 1: write out all four candidate mouths, then state which river survives.\n"
            "STEP 2: state the survivor's length using the RIVER ITSELF figure — the combined "
            "river-system figure measured through its longest tributary is NOT the river's length; "
            "say explicitly that you are rejecting it, and do not average the two.\n"
            "STEP 3: for EACH of the four further rivers write its check on its own line in the form "
            "'<river>: flows into <X> -> in the survivor's basin? yes/no; length=<val> km "
            f"(> {LENGTH_THRESHOLD:,} km? yes/no) -> counted? yes/no'. A river counts only when BOTH "
            "answers are yes; remember a river that flows into a TRIBUTARY of the survivor is still "
            "inside the survivor's basin.\n"
            "STEP 4: write the addition out explicitly on one line — the survivor's own length plus "
            "every counted river's length — and compute the total.\n"
            "Then report (a) that total in km as the keystone answer; (b) the survivor and all four "
            "candidate mouths; (c) the two conflicting lengths and which you used; (d) each of the "
            "four rivers' length, outflow and verdict; citing every source URL."
        ),
    }
