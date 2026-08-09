r"""
Test 149: Tier 5 (graph) — ARGMAX ACROSS INDEPENDENTLY 2-HOP-CHAINED BRANCHES (B).
Level: graph   Weight: long   Difficulty: 10/10

Second instance of the compound shape introduced by test 146, in an unrelated domain (146 is
hydro-geography; this one is observational astronomy) so the pair can be used as a within-shape
replication rather than a single lucky fixture. Same stack, different facts:

    per branch (2 hops, both page-only)
      HOP 1  observatory SITE  ->  the NAME of the LARGEST OPTICAL telescope installed there
      HOP 2  that telescope's article  ->  its PRIMARY MIRROR DIAMETER in metres
    then
      ARGMAX across the four branches -> which site's largest telescope has the biggest mirror.

  Hop 2's target page is unknown until hop 1 resolves that branch, so a flat one-round fan-out
  cannot reach the comparable values at all; a linear agent must serialise eight hops and hold four
  terminal values. Golden path = 7-8 visits (four site pages + three or four telescope pages — the
  Hooker telescope has no standalone article, it is a section of the Mount Wilson page), inside the
  suite's 6-9 visit budget.

Ground truth (verified against live English Wikipedia via the MediaWiki API, 2026-08-06):

  branch (given)                        HOP 1 largest telescope           HOP 2 primary mirror
  ------------------------------------  --------------------------------  --------------------
  Paranal Observatory (Chile)            Very Large Telescope (VLT)        8.2 m   [KEYSTONE]
      Paranal page: "The Very Large Telescope (VLT), the largest telescope on Paranal, is composed
      of four separate 8.2 m telescopes"; VLT infobox diameter: "4 x 8.2-metre Unit Telescopes"
  Palomar Observatory (California)       Hale Telescope                    5.08 m (200 inch; the
                                                                           pages also print 5.1 m)
  Kitt Peak National Observatory (AZ)    Nicholas U. Mayall Telescope      4.0 m (158 inch)
  Mount Wilson Observatory (California)  Hooker telescope                  2.54 m (100 inch)

  ARGMAX = Paranal Observatory / the VLT, 8.2 m primary mirror per unit telescope.
  MARGIN: 8.2 m vs the runner-up 5.08 m = +61% (a factor of 1.61); against the third and fourth
  branches it is 2.05x and 3.23x. The four diameters (2.54 / 4.0 / 5.08 / 8.2) are pairwise
  distinct with >25% between neighbours, so neither the argmax nor the coverage attribution can be
  flipped by one noisy extraction.

  NOTE ON THE PRINTED FIGURES (re-verified live 2026-08-07): 5.08 and 2.54 are the exact
  inch->metre conversions, but they are NOT the strings the articles print. The pages render
  "200-inch (5.1 m)" for the Hale and "100-inch (2.5 m)" for the Hooker (the Mount Wilson article
  also calls it "the 2.5 meter telescope" in prose). Both renderings are therefore accepted by the
  per-branch ``dia_rx``, so an agent that answers the task as asked — the diameter IN METRES — is
  credited for the value the page actually shows.

ADVERSARIAL CHECK ON HOP 1 (found while verifying, and mitigated): "the largest telescope at that
site" is NOT well defined at one of the four sites. Kitt Peak's article says only that "the largest
OPTICAL instruments at KPNO are the Mayall 4 meter telescope and the WIYN 3.5-meter", and separately
that "the ARO 12m Radio Telescope is also at the location" — so a careful, literal reading of an
unqualified "largest telescope" yields a 12 m radio dish, which is BIGGER than the winning branch's
8.2 m and silently flips the argmax. Both the task statement and the compiled plan therefore ask,
UNIFORMLY for all four branches (so the qualifier leaks nothing about which branch is affected), for
the largest OPTICAL telescope. That is the winning instrument at every site — Paranal's VLT,
Palomar's Hale, Kitt Peak's Mayall, Mount Wilson's Hooker — so the qualifier disambiguates without
narrowing anything, and at Kitt Peak it matches the article's own wording verbatim.

WHY IT DISCRIMINATES (the decoys are the historically famous instruments):
  * FAME DECOY 1 — the Hale Telescope at Palomar was the largest telescope in the world for 45
    years (1949-1993) and is the canonical "giant telescope" of popular science; it loses by 61%.
  * FAME DECOY 2 — the Hooker telescope at Mount Wilson (where the expansion of the universe was
    established) was the world's largest 1917-1949; it is the SMALLEST here.
  * SITE-vs-INSTRUMENT trap — a branch cannot be short-circuited from the site's name: no telescope
    here is named after its observatory (Paranal -> VLT, Palomar -> Hale, Kitt Peak -> Mayall,
    Mount Wilson -> Hooker), so hop 1 must actually be read.
  * MULTI-UNIT trap — the winning instrument is not a single dish: the VLT is four identical unit
    telescopes, so an agent may report a combined/equivalent aperture instead of ONE unit's
    8.2 m primary. The statement and the compiled plan both ask explicitly for one unit's mirror.

LEAK-SURFACE NOTE (honest): telescope apertures are better represented in parametric memory than
146's reservoir areas — a strong model may recall "the VLT is 8.2 m". The gate therefore also
requires GROUNDING (visit.count > 0), the un-gated coverage diagnostic is visit-capped so a
0-visit narration banks nothing, and the two most memorable instruments in the set are decoys that
lose. Treat this task as the replication partner of 146, not as its leak-resistance twin.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ── the four branches (single source of truth for statement, validators and the plan) ──
BRANCHES: List[Dict[str, Any]] = [
    {
        "key": "paranal",
        "site": "Paranal Observatory",
        "desc": "ESO's Paranal Observatory on Cerro Paranal in the Atacama Desert, Chile",
        "site_rx": r"paranal",
        "site_slug": r"wiki/paranal_observatory",
        "telescope": "Very Large Telescope (VLT)",
        "tel_rx": r"very\s+large\s+telescope|\bvlt\b",
        "tel_slug": r"wiki/very_large_telescope",
        "dia_rx": r"\b8\.2\b",
        "diameter_m": 8.2,
        "winner": True,
    },
    {
        "key": "palomar",
        "site": "Palomar Observatory",
        "desc": "Palomar Observatory on Palomar Mountain, California, USA",
        "site_rx": r"palomar",
        "site_slug": r"wiki/palomar_observatory",
        "telescope": "Hale Telescope",
        "tel_rx": r"\bhale\b",
        "tel_slug": r"wiki/hale_telescope",
        # 5.08 m, printed as "200-inch (5.1-meter)" on the pages — accept either rendering.
        "dia_rx": r"\b5\.0?8\b|\b5\.1\b|\b200[\s-]*(?:in\b|inch)",
        "diameter_m": 5.08,
        "winner": False,
    },
    {
        "key": "kittpeak",
        "site": "Kitt Peak National Observatory",
        "desc": "Kitt Peak National Observatory in the Quinlan Mountains, Arizona, USA",
        "site_rx": r"kitt\s*peak",
        "site_slug": r"wiki/kitt_peak_national_observatory",
        "telescope": "Nicholas U. Mayall Telescope",
        "tel_rx": r"mayall",
        "tel_slug": r"wiki/nicholas_u\._?_?mayall_telescope|wiki/mayall",
        # The metric alternative needs BOTH guards: ``\b`` alone treats the '.' in "2.4 m" as a word
        # boundary, so a bare ``\b4\s*m`` credits the MDM 2.4 m Hiltner (printed on the Kitt Peak
        # page) and any other "X.4 m". ``(?<![\d.])`` blocks a preceding digit or decimal point.
        "dia_rx": r"\b158[\s-]*(?:in\b|inch)|\bfour[\s-]*met|(?<![\d.])\b4(?:\.0)?\s*[-\s]*(?:m\b|met)",
        "diameter_m": 4.0,
        "winner": False,
    },
    {
        "key": "wilson",
        "site": "Mount Wilson Observatory",
        "desc": "Mount Wilson Observatory in the San Gabriel Mountains, California, USA",
        "site_rx": r"mount\s+wilson|mt\.?\s*wilson",
        "site_slug": r"wiki/mount_wilson_observatory",
        "telescope": "Hooker telescope",
        "tel_rx": r"hooker",
        # No standalone article: the Hooker is a section of the Mount Wilson Observatory page.
        "tel_slug": r"wiki/mount_wilson_observatory|wiki/hooker_telescope",
        # 2.54 m is the exact conversion, but the article prints "100-inch (2.5 m)" and calls it
        # "the 2.5 meter telescope" — so the metric value an agent reads off the page is 2.5.
        # ``\b2\.5\b`` alone cannot match "2.54" (5|4 is not a boundary), hence the optional digit.
        "dia_rx": r"\b2\.5(?:4)?\b|\b100[\s-]*(?:in\b|inch)",
        "diameter_m": 2.54,
        "winner": False,
    },
]

WINNER = next(b for b in BRANCHES if b["winner"])       # Paranal / the VLT
KEYSTONE_RX = re.compile(WINNER["dia_rx"], re.IGNORECASE)
_WINNER_NAME_RX = WINNER["tel_rx"] + "|" + WINNER["site_rx"]
_OTHERS_RX = "|".join(b["tel_rx"] + "|" + b["site_rx"] for b in BRANCHES if not b["winner"])

# TRUE superlatives only — words that assert a maximum. 'big'/'large' alone are not triggers (every
# instrument here is large, and "Very Large Telescope" is literally a name); 'top' is excluded
# because it collides with mountaintop/telescope-top vocabulary. Proximity windows use [^.;]
# (newline-tolerant, sentence-bounded) and may never cross another branch's name.
_SUP = r"largest|biggest|greatest|maximum|max\.?|widest|winner|answer"
_WINNER_WINS = re.compile(
    r"(?:" + _WINNER_NAME_RX + r")(?:(?!" + _OTHERS_RX + r")[^.;]){0,90}\b(?:" + _SUP + r")\b"
    r"|\b(?:" + _SUP + r")\b(?:(?!" + _OTHERS_RX + r")[^.;]){0,70}(?:" + _WINNER_NAME_RX + r")",
    re.IGNORECASE,
)
_OTHER_WINS = re.compile(
    r"(?:" + _OTHERS_RX + r")(?:(?!" + _WINNER_NAME_RX + r")[^.;]){0,90}\b(?:" + _SUP + r")\b"
    r"|\b(?:" + _SUP + r")\b(?:(?!" + _WINNER_NAME_RX + r")[^.;]){0,70}(?:" + _OTHERS_RX + r")",
    re.IGNORECASE,
)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "149",
        "test_name": "Tier 5: Argmax across four independently 2-hop-chained branches B (observatory -> largest telescope -> mirror diameter)",
        "difficulty_level": "10/10",
        "category": "Compound: per-branch 2-hop chain + cross-branch argmax",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {b['site']} — {b['desc']}" for i, b in enumerate(BRANCHES, 1))
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess from "
        "memory). Four astronomical observatory sites:\n"
        f"{listing}\n\n"
        "For EACH of the four sites you must follow a TWO-STEP chain — the page you need in step 2 "
        "is unknown until step 1 is resolved:\n"
        "  STEP 1 — open the observatory's own Wikipedia page and read the NAME of the LARGEST "
        "OPTICAL telescope installed at that site. Ignore radio telescopes, solar telescopes and "
        "interferometer arrays even where one of those has a wider dish — the comparison is between "
        "OPTICAL instruments. None of the four telescopes is named after its observatory, so you "
        "cannot infer this — you must read it.\n"
        "  STEP 2 — open THAT TELESCOPE's own article (or its section on the observatory page, if "
        "it has no separate article) and read the diameter of its PRIMARY MIRROR in metres. If the "
        "instrument consists of several identical unit telescopes, report the diameter of ONE "
        "unit's primary mirror, not a combined or equivalent aperture.\n\n"
        "Then COMPARE the four branches: determine which site's largest optical telescope has the "
        "BIGGEST primary mirror.\n\n"
        "Careful: two of these sites held the 'largest telescope in the world' record during the "
        "twentieth century — historical fame is not the criterion here; compare the four "
        "primary-mirror diameters you actually read.\n\n"
        "Report (a) the primary-mirror diameter in metres of the BIGGEST of the four, naming that "
        "telescope (and its site) and stating explicitly that it is the largest (this is the "
        "keystone answer); (b) for all four sites, the largest telescope there and its primary "
        "mirror diameter; and (c) the exact source URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The primary-mirror diameter (m) of the biggest of the four telescopes, naming it (the keystone)",
        "All four rows: observatory -> largest optical telescope -> primary mirror diameter",
        "Source URL for each observatory page and each telescope page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 5 pages visited (golden path 7-8: four observatory pages + the telescope pages)",
        "Resolves hop 1 for each branch (the largest OPTICAL telescope at each site — at Kitt Peak "
        "the wider ARO 12m RADIO dish is not the answer)",
        "Resolves hop 2 for each branch (each telescope's primary mirror diameter)",
        "Correctly names the VLT at Paranal (8.2 m per unit telescope) as the biggest — NOT the "
        "Hale Telescope (5.08 m, world's largest 1949-1993) and NOT the Hooker (2.54 m)",
        "Cites the observatory and telescope source pages",
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
    """KEYSTONE gate (hard 0/1): the primary answer names the winning branch (the VLT / Paranal) AS
    the biggest AND carries its 8.2 m mirror figure, AND the run is GROUNDED (visit.count > 0).

    Acceptance paths: (1) terse — only the winning branch is named; (2) explicit — a TRUE
    superlative bound to the winner inside a window that never crosses another branch's name.
    Vetoed when a different branch is asserted to be the biggest and the winner is not."""
    grounded = int((observability or {}).get("visit", {}).get("count", 0) or 0) > 0
    if not grounded:
        return False
    text = _primary_text(result)
    if not KEYSTONE_RX.search(text):
        return False
    if not re.search(_WINNER_NAME_RX, text, re.IGNORECASE):
        return False
    if _OTHER_WINS.search(text) and not _WINNER_WINS.search(text):
        return False
    if not re.search(_OTHERS_RX, text, re.IGNORECASE):
        return True                        # path 1: only the winner is named
    return bool(_WINNER_WINS.search(text))     # path 2: explicit superlative on the winner


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: golden path is 7-8 reads (four sites + their telescopes)."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 5, "score": min(1.0, n / 7.0),
            "reason": f"{n} visit(s) (target 7-8: four observatory pages + the telescope pages; >=5 to pass)"}


def validate_keystone_argmax(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the cross-branch argmax — the VLT at Paranal (8.2 m per unit telescope)
    has the biggest primary mirror of the four. The two twentieth-century record holders (Hale,
    Hooker) are the decoys a memory-anchored answer reaches for."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_argmax", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": ("Biggest primary mirror = the VLT at Paranal (8.2 m) reported" if passed
                       else "Argmax answer (VLT at Paranal, 8.2 m) missing/incorrect — beware the "
                            "Hale Telescope (5.08 m) and the Hooker telescope (2.54 m)")}


def validate_branch_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage/breadth diagnostic: how many of the FOUR branches were chained all the way
    through — the telescope's NAME (hop 1) and its mirror diameter (hop 2) both reported. This is
    the axis that separates a structured agent running four independent 2-hop chains from a linear
    one that resolves one branch and guesses; NOT gated on the keystone, so breadth survives a
    botched comparison. Visit-capped (``min(hits, n_visits)``, the canonical F29 pattern) because
    both facts are page-only."""
    text = _all_text(result)
    hits = [b["telescope"] for b in BRANCHES
            if re.search(b["tel_rx"], text, re.IGNORECASE)
            and re.search(b["dia_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(hits), n_visits)
    n = len(BRANCHES)
    return {"check": "branch_coverage", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} branches chained to telescope+mirror from visited pages "
                      f"({', '.join(hits[:credited]) or 'none'}; {len(hits)} text-matched, "
                      f"{n_visits} visit(s))"}


def validate_winner_chain(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the winning branch is spelled out end to end — the site (Paranal) and the
    instrument (the VLT). Short-circuits to 0 without the keystone."""
    if not _keystone_ok(result, observability):
        return {"check": "winner_chain", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> winning chain not credited"}
    text = _all_text(result)
    has_site = bool(re.search(WINNER["site_rx"], text, re.IGNORECASE))
    has_tel = bool(re.search(WINNER["tel_rx"], text, re.IGNORECASE))
    hits = int(has_site) + int(has_tel)
    return {"check": "winner_chain", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"winning branch: site(Paranal)={has_site}, telescope(VLT)={has_tel}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the pages the chains had to read (four observatory pages and the
    telescope pages). Short-circuits to 0 without the keystone."""
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    slugs = [b["site_slug"] for b in BRANCHES] + [b["tel_slug"] for b in BRANCHES]
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
    # None -> the harness applies its default structured rubric judge.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored CHAINED-BRANCH DAG scaffold for the ``graph_compiled`` variant.

    Two waves of FOUR — four independent 2-hop chains, then a merge (NOT a pure fan-out, NOT a
    single chain):
      * WAVE 1 — one leaf per GIVEN observatory, reading ONLY the NAME of the largest OPTICAL
        telescope there (hop 1). The "optical" qualifier is applied uniformly to all four leaves —
        it is what makes hop 1 well defined at every site (see the ARO 12m note above) and it
        cannot hint which site needed it.
      * WAVE 2 — one leaf per branch, each depending on ITS OWN wave-1 leaf and templating that
        result ({tel_*}) to read the primary-mirror diameter (hop 2), with the multi-unit
        instruction ("one unit's primary mirror") applied uniformly to every branch so the plan
        cannot hint which branch is multi-unit.
      * The ARGMAX lives only in the aggregation, which must write out all four
        observatory -> telescope -> diameter rows BEFORE concluding.

    Leak-free: names the four GIVEN sites and the GIVEN comparison attribute, but no telescope name,
    no diameter, and never which branch wins."""
    hop1 = [
        {
            "id": f"tel_{b['key']}",
            "instruction": (
                f"Open the English Wikipedia page for {b['site']} — {b['desc']}. Read ONLY the NAME "
                "of the LARGEST OPTICAL telescope installed at that site (the optical instrument "
                "with the biggest primary mirror there). Ignore radio telescopes, solar telescopes "
                "and interferometer arrays even where one of those has a wider dish. Report your "
                f"finding in the form '{b['site']}: largest optical telescope = <TELESCOPE NAME>' "
                "followed by the exact source URL. Do not report any diameter or other figure, and "
                "do not guess from memory."
            ),
            "expect": f"'{b['site']}: largest optical telescope = <TELESCOPE NAME>' — source URL",
            "depends_on": [],
        }
        for b in BRANCHES
    ]
    hop2 = [
        {
            "id": f"dia_{b['key']}",
            "instruction": (
                "The previous step identified the largest optical telescope at "
                f"{b['site']}: {{tel_{b['key']}}}. Open THAT TELESCOPE's own article (or, if it has "
                "no separate article, its section on the observatory's page) and read the diameter "
                "of its PRIMARY MIRROR in metres. If the instrument consists of several identical "
                "unit telescopes, report the diameter of ONE unit's primary mirror, not a combined "
                "or equivalent aperture. Report your finding in the form '<TELESCOPE NAME>: primary "
                "mirror = <number> m' followed by the exact source URL, so the instrument's name "
                "stays attached to its figure. Do not guess from memory."
            ),
            "expect": "'<TELESCOPE NAME>: primary mirror = <number> m' — source URL",
            "depends_on": [f"tel_{b['key']}"],
        }
        for b in BRANCHES
    ]
    return {
        "leaves": hop1 + hop2,
        "aggregation": (
            "You now have, for each of the four observatory sites, the largest optical telescope "
            "there and its primary-mirror diameter. FIRST write out all four rows explicitly, "
            "one per line, in the form '<observatory> -> <telescope> -> <primary mirror> m'. THEN "
            "compare the four diameters and identify the telescope with the BIGGEST primary mirror "
            "— compare ONE unit's mirror diameter, never a combined or equivalent aperture. Report "
            "(a) that biggest primary-mirror diameter in metres, naming the telescope and its site "
            "and stating explicitly that it is the largest of the four — this is the keystone "
            "answer; (b) all four observatory -> telescope -> diameter rows; and (c) every source "
            "URL you used."
        ),
    }
