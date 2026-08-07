r"""
Test 146: Tier 5 (graph) — ARGMAX ACROSS INDEPENDENTLY 2-HOP-CHAINED BRANCHES.
Level: graph   Weight: long   Difficulty: 10/10

COMPOUND SHAPE (two axis TYPES stacked, not one axis made deeper): every existing argmax task in
the suite compares a value that is readable on ONE page (062 prominence, 091 dam heights, 084 lake
depths), and every existing chain task (050/051/065/092/095…) terminates in a SINGLE answer with no
comparison. This task requires FOUR INDEPENDENT 2-HOP CHAINS and then an ARGMAX across their four
terminal values. Neither axis alone solves it:

    per branch (2 hops, both page-only)
      HOP 1  hydroelectric project  ->  the NAME of the reservoir it impounds   (infobox 'Reservoir')
      HOP 2  that reservoir's page  ->  its SURFACE AREA in km²                 (infobox 'Surface area')
    then
      ARGMAX across the four branches' terminal areas -> which reservoir is the largest by AREA.

  A linear agent must serialize eight dependent hops and hold four terminal values in one degrading
  scratchpad; a flat fan-out agent cannot fetch hop 2 at all, because the page to open in hop 2 is
  unknown until hop 1 resolves that branch. Golden path = 8 visits (4 projects + 4 reservoirs),
  inside the suite's 6-9 visit budget.

Ground truth (verified against live English Wikipedia via the MediaWiki API, 2026-08-06 — each
project's infobox 'Reservoir created' / 'res_name' + 'res_surface' rows, cross-checked against the
reservoir's OWN article infobox 'Surface area' / 'area' row):

  branch (given)                        HOP 1 reservoir        HOP 2 surface area        capacity
  ------------------------------------  ---------------------  ------------------------  ---------
  Churchill Falls Generating Station     Smallwood Reservoir    6,527 km²  [KEYSTONE]      32.64 km³
    (station page quotes 6,988 km² for the same reservoir; the reservoir's own page says 6,527 km²
     — both are accepted, see KEYSTONE_RX; the argmax is unaffected because either value wins)
  Daniel-Johnson Dam                     Manicouagan Reservoir  1,942 km² (dam page 1,950)  137.9 km³
  W. A. C. Bennett Dam                   Williston Lake         1,761 km²                   74 km³
  Hoover Dam                             Lake Mead                640 km² (247 sq mi)      ~32 km³

  ARGMAX (surface area) = Smallwood Reservoir, 6,527 km².
  MARGIN: 6,527 vs the runner-up 1,942 km² = 3.36x (+236%). Even the two figures that circulate for
  Smallwood (6,527 / 6,988) and for Manicouagan (1,942 / 1,950) cannot come within a factor of 3 of
  each other, so no single noisy extraction can flip the argmax. The four areas are pairwise
  distinct by more than an order of magnitude spread (640 / 1,761 / 1,942 / 6,527), so the coverage
  diagnostic is collision-free.

WHY IT DISCRIMINATES (the decoys are engineered, not incidental):
  * CAPACITY DECOY (the ranking flips on the other axis): by water VOLUME the winner is a DIFFERENT
    branch — Manicouagan holds 137.9 km³ against Smallwood's 32.64 km³ (4.2x the volume in a third
    of the area). An agent that grabs "the largest reservoir" without holding onto the requested
    attribute (SURFACE AREA) lands on Manicouagan.
  * FAME DECOY: Hoover Dam is by far the most famous project in the set and its Lake Mead is
    routinely described as "the largest reservoir in the United States" (by volume) — yet it is the
    SMALLEST of the four by surface area (640 km², one tenth of the winner).
  * NAME-BREAK: no reservoir here shares its project's name (Churchill Falls -> Smallwood,
    Daniel-Johnson -> Manicouagan, W. A. C. Bennett -> Williston, Hoover -> Mead), so hop 1 cannot
    be short-circuited by echoing the given project name — the page must actually be read.

ADVERSARIAL CHECK ON HOP 1 (found while verifying, and mitigated): one of the four infoboxes lists
TWO impoundments in its 'Reservoir created' row (a principal reservoir plus a much smaller secondary
one), so a hop-1 read could return the wrong body of water and collapse that branch. Both the task
statement and the compiled plan therefore say, UNIFORMLY for all four branches (so the instruction
leaks nothing about which branch is affected), that where more than one reservoir is listed the
PRINCIPAL (largest) one is meant. The remaining three infoboxes name exactly one reservoir each.

LEAK RESISTANCE: the keystone is the surface area of the SMALLWOOD RESERVOIR — an obscure figure
for an obscure impoundment (88 dikes in Labrador, no single large dam) that no consumer model
recalls parametrically; the memorable superlatives in this domain ("largest reservoir in the US",
"largest man-made lake") point at the decoys. \b6,?527\b / \b6,?988\b collide with none of the other
figures on any page in the task (areas 640 / 1,761 / 1,942 / 1,950, volumes 32.64 / 74 / 137.9,
capacities 2,078.8 / 2,907 / 5,428 MW).
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ── the four branches (single source of truth for statement, validators and the plan) ──
# ``project_*`` is the GIVEN (branch entry point); ``res_*``/``area_rx`` are the two page-only facts
# the branch must chain to. Nothing numeric is leaked into the task statement or the compiled plan.
BRANCHES: List[Dict[str, Any]] = [
    {
        "key": "churchill",
        "project": "Churchill Falls Generating Station",
        "desc": "the Churchill Falls hydroelectric generating station in Labrador, Canada",
        "project_rx": r"churchill\s+falls",
        "project_slug": r"wiki/churchill_falls",
        "reservoir": "Smallwood Reservoir",
        "res_rx": r"smallwood",
        "res_slug": r"wiki/smallwood_reservoir",
        # 6,527 km² (reservoir's own page) or 6,988 km² (the station page's figure for the same
        # reservoir); the imperial equivalents 2,520 / 2,698 sq mi are guarded by a 'sq'/'square'.
        "area_rx": r"\b6[,.\s]?527\b|\b6[,.\s]?988\b|\b2[,.\s]?520\s*(?:sq|square)|\b2[,.\s]?698\s*(?:sq|square)",
        "area_km2": 6527,
        "winner": True,
    },
    {
        # ``key`` is keyed on the GIVEN project, never on the reservoir: it becomes the compiled
        # plan's leaf id ("res_<key>"/"area_<key>"), which must not spell out hop 1's answer.
        "key": "danieljohnson",
        "project": "Daniel-Johnson Dam",
        # NB: the description must not name the river this dam sits on — that river shares its name
        # with the reservoir, i.e. it would leak hop 1's answer into the task statement AND the
        # compiled plan (both render ``desc``). Located by region instead.
        "desc": "the Daniel-Johnson Dam in the Côte-Nord region of Quebec, Canada",
        "project_rx": r"daniel[-\s]?johnson|manic[-\s]?5",
        "project_slug": r"wiki/daniel-johnson_dam",
        "reservoir": "Manicouagan Reservoir",
        "res_rx": r"manicouagan\s+reservoir|lake\s+manicouagan",
        "res_slug": r"wiki/manicouagan_reservoir",
        # 1,942 km² (reservoir page) or 1,950 km² (dam page) — the two circulating figures ONLY.
        # A ``9[45]\d`` tail would also swallow 1,940-1,959, i.e. eight plausible 20th-century
        # years: the dam's own page prints 1959 six times (construction_began), plus 1945/1955/1956.
        # That let "Manicouagan Reservoir ... construction began 1959" bank this branch's AREA in
        # the un-gated breadth diagnostic without the figure ever being read.
        "area_rx": r"\b1[,.\s]?9(?:42|50)\b",
        "area_km2": 1942,
        "winner": False,
    },
    {
        "key": "bennett",
        "project": "W. A. C. Bennett Dam",
        "desc": "the W. A. C. Bennett Dam on the Peace River, British Columbia, Canada",
        "project_rx": r"bennett\s+dam|w\.?\s*a\.?\s*c\.?\s*bennett",
        "project_slug": r"wiki/w\._?a\._?c\._?bennett_dam",
        "reservoir": "Williston Lake",
        "res_rx": r"williston",
        "res_slug": r"wiki/williston_lake",
        "area_rx": r"\b1[,.\s]?761\b",
        "area_km2": 1761,
        "winner": False,
    },
    {
        "key": "hoover",
        "project": "Hoover Dam",
        "desc": "the Hoover Dam on the Colorado River, on the Nevada/Arizona border, USA",
        "project_rx": r"hoover\s+dam",
        "project_slug": r"wiki/hoover_dam",
        "reservoir": "Lake Mead",
        "res_rx": r"\bmead\b",
        "res_slug": r"wiki/lake_mead",
        "area_rx": r"\b640\b|\b247\s*(?:sq|square|mi)",
        "area_km2": 640,
        "winner": False,
    },
]

WINNER = next(b for b in BRANCHES if b["winner"])          # Smallwood Reservoir
_OTHERS_RX = "|".join(b["res_rx"] for b in BRANCHES if not b["winner"])
KEYSTONE_RX = re.compile(WINNER["area_rx"], re.IGNORECASE)

# TRUE superlatives only — words that actually assert "this is the maximum". 'large'/'big' alone
# are NOT triggers (every reservoir here is large), and 'top' is excluded because it collides with
# dam vocabulary ("top of the dam", "crest top"); comparatives ("exceeds", "greater than") are not
# triggers either. Proximity windows use [^.;] (newline-tolerant, sentence-bounded) and may never
# cross another reservoir's name, so a listing sentence cannot cross-credit.
_SUP = r"largest|biggest|greatest|maximum|max\.?|widest|winner|answer"
_WINNER_WINS = re.compile(
    WINNER["res_rx"] + r"(?:(?!" + _OTHERS_RX + r")[^.;]){0,90}\b(?:" + _SUP + r")\b"
    r"|\b(?:" + _SUP + r")\b(?:(?!" + _OTHERS_RX + r")[^.;]){0,70}" + WINNER["res_rx"],
    re.IGNORECASE,
)
_OTHER_WINS = re.compile(
    r"(?:" + _OTHERS_RX + r")(?:(?!" + WINNER["res_rx"] + r")[^.;]){0,90}\b(?:" + _SUP + r")\b"
    r"|\b(?:" + _SUP + r")\b(?:(?!" + WINNER["res_rx"] + r")[^.;]){0,70}(?:" + _OTHERS_RX + r")",
    re.IGNORECASE,
)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "146",
        "test_name": "Tier 5: Argmax across four independently 2-hop-chained branches (hydro project -> reservoir -> surface area)",
        "difficulty_level": "10/10",
        "category": "Compound: per-branch 2-hop chain + cross-branch argmax",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {b['project']} — {b['desc']}" for i, b in enumerate(BRANCHES, 1))
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess from "
        "memory). Four large hydroelectric projects in North America:\n"
        f"{listing}\n\n"
        "For EACH of the four projects you must follow a TWO-STEP chain — the page you need in step "
        "2 is unknown until step 1 is resolved:\n"
        "  STEP 1 — open the project's own Wikipedia page and read the NAME of the reservoir it "
        "impounds (the infobox 'Reservoir created' / 'Reservoir' row). None of the four reservoirs "
        "is named after its project, so you cannot infer this — you must read it. If a page lists "
        "more than one impoundment, take the PRINCIPAL (largest) one.\n"
        "  STEP 2 — open THAT RESERVOIR's own Wikipedia page and read its SURFACE AREA in square "
        "kilometres (the infobox 'Surface area' row).\n\n"
        "Then COMPARE the four branches: determine which of the four reservoirs has the LARGEST "
        "SURFACE AREA.\n\n"
        "Careful: the question is about surface AREA, not stored water VOLUME — the ranking by "
        "volume is a different ranking, and fame is not a guide here; compare the four surface "
        "areas you actually read.\n\n"
        "Report (a) the surface area in km² of the LARGEST-BY-AREA reservoir, naming that reservoir "
        "and stating explicitly that it is the largest (this is the keystone answer); (b) for all "
        "four projects, the reservoir each impounds and that reservoir's surface area; and (c) the "
        "exact source URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The surface area (km²) of the largest-by-area reservoir, naming it (the keystone)",
        "All four rows: project -> reservoir -> surface area",
        "Source URL for each project page and each reservoir page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 5 pages visited (golden path 8: four project pages + four reservoir pages)",
        "Resolves hop 1 for each branch (the reservoir each project impounds)",
        "Resolves hop 2 for each branch (each reservoir's surface area)",
        "Correctly names the Smallwood Reservoir as the largest by surface area (~6,527 km²), NOT "
        "the Manicouagan Reservoir (larger by VOLUME) and NOT Lake Mead (the famous one)",
        "Cites the project and reservoir source pages",
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


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE gate (hard 0/1): the primary answer names the Smallwood Reservoir AS the
    largest-by-area AND carries its surface-area figure, AND the run is GROUNDED (visit.count > 0,
    so a correct-but-ungrounded parametric guess earns nothing).

    Acceptance paths: (1) terse — only the winning reservoir is named; (2) explicit — a TRUE
    superlative bound to the winner inside a window that never crosses another reservoir's name.
    Vetoed when a different reservoir is asserted to be the largest and the winner is not."""
    grounded = int((observability or {}).get("visit", {}).get("count", 0) or 0) > 0
    if not grounded:
        return False
    text = _primary_text(result)
    if not KEYSTONE_RX.search(text):
        return False
    if not re.search(WINNER["res_rx"], text, re.IGNORECASE):
        return False
    if _OTHER_WINS.search(text) and not _WINNER_WINS.search(text):
        return False
    if not re.search(_OTHERS_RX, text, re.IGNORECASE):
        return True                      # path 1: only the winner is named
    return bool(_WINNER_WINS.search(text))   # path 2: explicit superlative on the winner


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: the golden path is 8 reads (4 projects + 4 reservoirs)."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 5, "score": min(1.0, n / 8.0),
            "reason": f"{n} visit(s) (target 8: four project pages + four reservoir pages; >=5 to pass)"}


def validate_keystone_argmax(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the cross-branch argmax — the Smallwood Reservoir (~6,527 km²) is the
    largest of the four by SURFACE AREA. Reachable only by completing at least the winning branch's
    two hops and comparing it against the others; the volume ranking (Manicouagan) and the fame
    anchor (Lake Mead) both produce a different, wrong answer."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_argmax", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": ("Largest-by-area reservoir = Smallwood Reservoir (~6,527 km²) reported" if passed
                       else "Argmax answer (Smallwood Reservoir, ~6,527 km²) missing/incorrect — beware "
                            "the Manicouagan Reservoir (largest by VOLUME) and Lake Mead (most famous)")}


def validate_branch_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage/breadth diagnostic: how many of the FOUR branches were chained all the way
    through — i.e. the reservoir's NAME (hop 1) and its surface-area figure (hop 2) are both
    reported. This is the axis that separates a structured agent (which runs four independent
    2-hop chains and holds all four terminal values) from a linear one that resolves one or two
    branches and guesses. Deliberately NOT gated on the keystone — breadth is credited even when
    the final comparison is botched.

    Visit-capped (``min(hits, n_visits)``, the canonical F29 pattern): both facts are page-only, so
    a 0-visit run that narrates them from parametric memory banks nothing."""
    text = _all_text(result)
    hits = [b["reservoir"] for b in BRANCHES
            if re.search(b["res_rx"], text, re.IGNORECASE)
            and re.search(b["area_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(hits), n_visits)
    n = len(BRANCHES)
    return {"check": "branch_coverage", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} branches chained to reservoir+area from visited pages "
                      f"({', '.join(hits[:credited]) or 'none'}; {len(hits)} text-matched, "
                      f"{n_visits} visit(s))"}


def validate_winner_chain(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the WINNING branch is spelled out end to end — the project (Churchill Falls)
    and the reservoir it impounds (Smallwood). Short-circuits to 0 without the keystone, so a wrong
    argmax cannot bank chain credit."""
    if not _keystone_ok(result, observability):
        return {"check": "winner_chain", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> winning chain not credited"}
    text = _all_text(result)
    has_project = bool(re.search(WINNER["project_rx"], text, re.IGNORECASE))
    has_res = bool(re.search(WINNER["res_rx"], text, re.IGNORECASE))
    hits = int(has_project) + int(has_res)
    return {"check": "winner_chain", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"winning branch: project(Churchill Falls)={has_project}, reservoir(Smallwood)={has_res}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the pages the chains had to read (8 candidate slugs — four project
    pages and four reservoir pages). Short-circuits to 0 without the keystone."""
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    slugs = [b["project_slug"] for b in BRANCHES] + [b["res_slug"] for b in BRANCHES]
    cited = sum(1 for s in slugs if re.search(s, text))
    return {"check": "citations", "passed": cited >= 3, "score": min(1.0, cited / 5.0),
            "reason": f"{cited}/8 source page(s) cited (need >=3; target >=5)"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_argmax,
        validate_branch_coverage,
        validate_winner_chain,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge, as in the sibling graph tasks.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored CHAINED-BRANCH DAG scaffold for the ``graph_compiled`` variant.

    Two waves of FOUR, a genuine branch-and-chain tree (NOT a pure fan-out, NOT a single chain):
      * WAVE 1 — four INDEPENDENT leaves, one per GIVEN project, each reading ONLY the NAME of the
        reservoir that project impounds (hop 1).
      * WAVE 2 — four INDEPENDENT leaves, each depending on ITS OWN branch's wave-1 leaf and
        templating that leaf's result ({res_*}) to open the reservoir's page and read the surface
        area (hop 2). The four chains never touch each other, which is exactly why a runtime planner
        struggles: hop 2's target is unknowable before hop 1 resolves.
      * The ARGMAX itself lives only in the aggregation, which must write out all four
        project -> reservoir -> area rows BEFORE concluding.

    Leak-free: it names the four GIVEN projects and the GIVEN comparison attribute (surface area),
    but no reservoir name, no area, no volume, and never which branch wins."""
    hop1 = [
        {
            "id": f"res_{b['key']}",
            "instruction": (
                f"Open the English Wikipedia page for {b['project']} — {b['desc']}. Read ONLY the "
                "NAME of the reservoir (the body of water) that this project impounds, from the "
                "infobox 'Reservoir created' / 'Reservoir' row; if more than one impoundment is "
                "listed there, take the PRINCIPAL (largest) one. Report your finding in the form "
                f"'{b['project']} impounds: <RESERVOIR NAME>' followed by the exact source URL. Do "
                "not report any area, volume or capacity figure, and do not guess from memory."
            ),
            "expect": f"'{b['project']} impounds: <RESERVOIR NAME>' — source URL",
            "depends_on": [],
        }
        for b in BRANCHES
    ]
    hop2 = [
        {
            "id": f"area_{b['key']}",
            "instruction": (
                "The previous step identified the reservoir impounded by "
                f"{b['project']}: {{res_{b['key']}}}. Open THAT RESERVOIR's own English Wikipedia "
                "page and read its SURFACE AREA in square kilometres from the infobox 'Surface "
                "area' row. Report your finding in the form '<RESERVOIR NAME>: surface area = "
                "<number> km²' followed by the exact source URL, so the reservoir's name stays "
                "attached to its figure. Do not report its water volume/capacity, and do not guess "
                "from memory."
            ),
            "expect": "'<RESERVOIR NAME>: surface area = <number> km²' — source URL",
            "depends_on": [f"res_{b['key']}"],
        }
        for b in BRANCHES
    ]
    return {
        "leaves": hop1 + hop2,
        "aggregation": (
            "You now have, for each of the four hydroelectric projects, the reservoir it impounds "
            "and that reservoir's surface area in km². FIRST write out all four rows explicitly, "
            "one per line, in the form '<project> -> <reservoir> -> <surface area> km²'. THEN "
            "compare the four surface areas and identify the reservoir with the LARGEST surface "
            "area — compare AREA, not stored water volume. Report (a) that largest reservoir's "
            "surface area in km², naming the reservoir and stating explicitly that it is the "
            "largest of the four — this is the keystone answer; (b) all four "
            "project -> reservoir -> area rows; and (c) every source URL you used."
        ),
    }
