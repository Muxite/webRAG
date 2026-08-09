r"""
Test 126: Tier 5 (adaptive_targeted) — BRANCH-TO-ELIMINATE (survivor). Bucket A.
Level: graph   Weight: long   Difficulty: 9/10

LOW-CONTEXT DECISION-FULCRUM task for a GOOD ADAPTIVE AGENT: a disciplined interleaved
plan->act->observe->decide loop must check EACH candidate with one quick read and NOT shortcut to
the famous guess. Golden path = 3-4 precise visits, not breadth.

    DECISION (the fulcrum)
      "First cartridge-based handheld game console" fame-anchors on the Nintendo Game Boy (1989) — but
      the Milton Bradley Microvision (1979) was the FIRST handheld console with interchangeable ROM
      cartridges, a full decade earlier. The Atari Lynx and Sega Game Gear are also cartridge
      handhelds but arrived in 1989-1990. Exactly ONE was first: the Microvision. Resolving it
      requires reading each console's release year, not equating "handheld with cartridges" with the
      famous Game Boy.

    KEYSTONE (leak-resistant attribute of the survivor)
      Read the survivor's LCD screen resolution in pixels directly from its page.

Ground truth (verified against live English Wikipedia, 2026-07-10):

  Candidates — release / status:
  ┌───────────────────────────────┬──────────────────────────────────────────────┬────────────┐
  │ Nintendo Game Boy (fame decoy)│ cartridge handheld, released 1989              │ eliminated │
  │ Atari Lynx                    │ cartridge handheld, released 1989 (colour)     │ eliminated │
  │ Sega Game Gear                │ cartridge handheld, released 1990 (colour)     │ eliminated │
  │ Milton Bradley Microvision ← S│ FIRST handheld console with interchangeable    │ SURVIVES  │
  │                               │ ROM cartridges, released November 1979         │            │
  └───────────────────────────────┴──────────────────────────────────────────────┴────────────┘
      Microvision "the first handheld game console that used interchangeable ROM cartridges",
      released November 1979 — a decade before the Game Boy.

  Keystone (survivor attribute):
      Microvision — LCD screen resolution 16 x 16 pixels.  [KEYSTONE = 16x16 pixels]

Why leak-resistant: the 16x16 pixel resolution is a page-only spec; \b16\s*[x×]\s*16\b collides with
none of the decoys' resolutions (Game Boy 160x144; Lynx 160x102; Game Gear 160x144), so electing the
famous Game Boy — or naming the Microvision without reading its page — cannot produce it.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text, dimension_value_matches


CANDIDATES: List[Dict[str, Any]] = [
    {
        "key": "gameboy", "name": "Nintendo Game Boy",
        "desc": "the Nintendo Game Boy",
        "name_rx": r"game\s*boy", "prop_rx": r"1989|nintendo",
        "slug_rx": r"wiki/game_boy", "survivor": False,
    },
    {
        "key": "lynx", "name": "Atari Lynx",
        "desc": "the Atari Lynx",
        "name_rx": r"lynx", "prop_rx": r"1989|atari|colou?r",
        "slug_rx": r"wiki/atari_lynx", "survivor": False,
    },
    {
        "key": "gamegear", "name": "Sega Game Gear",
        "desc": "the Sega Game Gear",
        "name_rx": r"game\s*gear", "prop_rx": r"1990|sega|colou?r",
        "slug_rx": r"wiki/game_gear", "survivor": False,
    },
    {
        "key": "microvision", "name": "Milton Bradley Microvision",
        "desc": "the Milton Bradley Microvision",
        "name_rx": r"microvision", "prop_rx": r"1979|first|interchangeable|cartridge|milton bradley",
        "slug_rx": r"wiki/microvision", "survivor": True,
    },
]
SURVIVOR = next(c for c in CANDIDATES if c["survivor"])  # Milton Bradley Microvision

# ── keystone: Microvision LCD resolution, 16 x 16 pixels ──
# Joiner-tolerant via ``dimension_value_matches`` below: accepts "16x16" / "16 x 16" / "16×16"
# AND the spelled-out "16 by 16" — a correctly grounded answer must not false-fail merely for
# using a word instead of the "x"/"×" operator.


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "126",
        "test_name": "Tier 5 targeted: survivor (first cartridge-based handheld console -> LCD resolution)",
        "difficulty_level": "9/10",
        "category": "adaptive_targeted",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {c['name']} — {c['desc']}" for i, c in enumerate(CANDIDATES, 1))
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess from "
        "memory). Two stages; the second stage's target is unknown until the first is resolved. Be "
        "disciplined: one quick check per candidate, do NOT shortcut to the famous one.\n\n"
        "STAGE 1 — eliminate to one survivor. Four handheld game consoles that use interchangeable "
        "cartridges:\n"
        f"{listing}\n"
        "Exactly ONE of these was the FIRST handheld game console with interchangeable ROM "
        "cartridges. Open EACH console's page and read its RELEASE YEAR: three of them were released "
        "in 1989-1990, while one appeared a full decade earlier. Determine the release year of all "
        "four; do NOT simply pick the most famous cartridge handheld.\n\n"
        "STAGE 2 — read the keystone. Open the surviving console's page and read its LCD SCREEN "
        "RESOLUTION in pixels, directly from the page.\n\n"
        "Report: (a) the survivor's LCD resolution in pixels (this single figure is the keystone "
        "answer); (b) which of the four was the survivor and each one's release year; citing the "
        "exact Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The survivor console's LCD screen resolution in pixels (the leak-resistant keystone)",
        "Which console was the first cartridge-based handheld (the survivor) + each candidate's release year",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 3 pages visited (candidates + the survivor); low-context, no breadth reward",
        "Determines the release year of ALL FOUR consoles (branch-to-eliminate)",
        "Correctly elects the Milton Bradley Microvision as the first cartridge handheld (not the famous Game Boy)",
        "Reports the survivor's LCD resolution (16 x 16 pixels)",
        "Cites the survivor page (Microvision)",
    ]


def _primary_text(result: Dict[str, Any]) -> str:
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list) and deliv and deliv[0] is not None:
            return str(deliv[0])
    return extract_final_text(result)


def _all_text(result: Dict[str, Any]) -> str:
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return " ".join(parts)


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    grounded = int((observability or {}).get("visit", {}).get("count", 0) or 0) > 0
    return grounded and dimension_value_matches(_primary_text(result), 16, 16)


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 3, "score": min(1.0, n / 4.0),
            "reason": f"{n} visit(s) (low-context target 3-4: candidates + survivor)"}


def validate_keystone_resolution(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_resolution", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "LCD resolution 16 x 16 pixels present" if passed
                      else "Keystone LCD resolution (16 x 16 pixels, Microvision) missing/incorrect"}


def validate_branch_exploration(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the FOUR consoles the agent resolved (named + gave its
    release year/status). Visit-capped; NOT gated on the keystone."""
    text = _all_text(result)
    text_hits = [c["name"] for c in CANDIDATES
                 if re.search(c["name_rx"], text, re.IGNORECASE) and re.search(c["prop_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(text_hits), n_visits)
    n = len(CANDIDATES)
    return {"check": "branch_exploration", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} consoles resolved from visited pages "
                      f"({', '.join(text_hits[:credited]) or 'none'}; {len(text_hits)} text-matched, {n_visits} visit(s))"}


def validate_survivor(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result, observability):
        return {"check": "survivor", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor identification not credited"}
    has = bool(re.search(SURVIVOR["name_rx"], _all_text(result), re.IGNORECASE))
    return {"check": "survivor", "passed": has, "score": 1.0 if has else 0.0,
            "reason": f"survivor (Microvision) named={has}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for c in CANDIDATES if re.search(c["slug_rx"], text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2: e.g. survivor + one eliminated)"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_resolution, validate_branch_exploration,
            validate_survivor, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG scaffold. Two waves (fan-out of 4 -> 1 chain leaf).
    STRUCTURE only — names the GIVEN candidates and the GIVEN 'first cartridge-based handheld'
    criterion but leaks NO verdict, NOT which console survives, and NOT the LCD resolution."""
    cand_leaves = [
        {
            "id": f"cand_{c['key']}",
            "instruction": (
                f"Open the Wikipedia page for {c['name']} — {c['desc']}. Read its RELEASE YEAR (and "
                "confirm it is a handheld game console that uses interchangeable cartridges). Report "
                f"the console's name ({c['name']}), its release year, and the exact Wikipedia URL. Do "
                "not guess from memory; report no other fact."
            ),
            "expect": f"{c['name']} — its release year — source URL",
            "depends_on": [],
        }
        for c in CANDIDATES
    ]
    survivor_leaf = {
        "id": "survivor_resolution",
        "instruction": (
            "You are given the four candidate consoles and each one's release year:\n"
            "  Nintendo Game Boy -> {cand_gameboy}\n"
            "  Atari Lynx -> {cand_lynx}\n"
            "  Sega Game Gear -> {cand_gamegear}\n"
            "  Milton Bradley Microvision -> {cand_microvision}\n"
            "Determine which SINGLE one was the FIRST handheld game console with interchangeable ROM "
            "cartridges (released a full decade before the others). Open THAT surviving console's "
            "Wikipedia page and read its LCD SCREEN RESOLUTION in pixels. Report the surviving "
            "console, its LCD resolution in pixels, and the exact source URL. Do not guess from memory."
        ),
        "expect": "SURVIVING (first cartridge handheld) console + its LCD resolution in pixels — source URL",
        "depends_on": [f"cand_{c['key']}" for c in CANDIDATES],
    }
    return {
        "leaves": cand_leaves + [survivor_leaf],
        "aggregation": (
            "You now have (1) each console's release year and (2) which single one was the first "
            "cartridge-based handheld (the survivor) and its LCD resolution. Write out all four "
            "release years BEFORE concluding which survives. Then report (a) the survivor's LCD "
            "resolution in pixels — this single figure is the keystone answer; (b) which console was "
            "the survivor and each one's release year; citing every source URL."
        ),
    }
