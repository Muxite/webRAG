"""
Test 162: Tier 5 (graph) — SEQUENTIAL PREFIX -> 7-WAY FAN-OUT -> ARGMAX MERGE.
Level: graph   Weight: long   Difficulty: 9/10   Category: sequential_prefix_fan_out

The suite's breadth tasks (152-158) are all sets of INDEPENDENT one-hop page reads: the candidate
set is handed to the agent in the mandate. This task is the missing shape — the candidate set is
NOT namable from the prompt and has to be *discovered* by a genuinely dependent chain first, after
which each discovered member needs its own multi-step subtask, and only the aggregate over ALL of
them answers the question.

    PHASE 1 — SEQUENTIAL PREFIX (3 dependent hops; each target unknowable before the prior read)
      HOP 1 (given start)  Hubble instrument "High Speed Photometer" -> WHAT corrective-optics
                           package replaced it.                                  [intermediate]
      HOP 2 (from hop 1)   That package's page -> WHICH Space Shuttle mission installed it, and
                           the mission's exact launch date.                       [intermediate]
      HOP 3 (from hop 2)   The mission's page -> its full FLIGHT CREW (7 people).  [candidate set]

    PHASE 2 — FAN-OUT (7 mutually independent branches, ~2-3 actions each)
      For each crew member: locate that astronaut's own page, read the date of birth, then DERIVE
      the age in whole years on the mission's launch date. No branch consumes another's output, so
      branch order is worth nothing.

    PHASE 3 — MERGE (argmax over all 7)
      Who was the OLDEST at launch, and how old. Skipping or fabricating any branch breaks it.

Ground truth (verified against live English Wikipedia, 2026-08-29):
  - High Speed Photometer (https://en.wikipedia.org/wiki/High_Speed_Photometer):
    "During the first servicing mission, in December 1993, it was replaced by the Corrective
    Optics Space Telescope Axial Replacement (COSTAR)".
  - COSTAR (https://en.wikipedia.org/wiki/COSTAR): "It was flown via shuttle to the telescope in
    the servicing mission STS-61, on December 2, 1993, and successfully installed over a period of
    eleven days."
  - STS-61 (https://en.wikipedia.org/wiki/STS-61): infobox launch "December 2, 1993, 09:27:00 UTC";
    crew = Richard O. Covey (CDR), Kenneth Bowersox (PLT), Kathryn C. Thornton, Claude Nicollier,
    Jeffrey A. Hoffman, Story Musgrave, Thomas D. Akers (MS1-MS5).

  Branch table — date of birth (infobox, verbatim) -> age in whole years on 1993-12-02:
    Story Musgrave    https://en.wikipedia.org/wiki/Story_Musgrave       1935-08-19 -> 58  KEYSTONE
    Claude Nicollier  https://en.wikipedia.org/wiki/Claude_Nicollier     1944-09-02 -> 49
    Jeffrey A. Hoffman https://en.wikipedia.org/wiki/Jeffrey_A._Hoffman  1944-11-02 -> 49
    Richard O. Covey  https://en.wikipedia.org/wiki/Richard_O._Covey     1946-08-01 -> 47
    Thomas D. Akers   https://en.wikipedia.org/wiki/Thomas_D._Akers      1951-05-20 -> 42
    Kathryn C. Thornton https://en.wikipedia.org/wiki/Kathryn_C._Thornton 1952-08-17 -> 41
    Kenneth Bowersox  https://en.wikipedia.org/wiki/Kenneth_Bowersox     1956-11-14 -> 37

Keystone margin: Musgrave (58) beats the joint runner-up (Nicollier / Hoffman, 49) by NINE years.
Every one of the seven birthdays falls BEFORE 2 December, so naive year-subtraction (1993 - YOB)
and exact date arithmetic agree on all seven ages — there is no off-by-one band in which the
argmax could flip, and a single noisy extraction would have to be wrong by ~9 years to change the
answer. Decoys a stop-early agent produces are far away: the launch year 1993, the eleven-day
installation, and Hubble's own 1990 deployment share no token with 58.

Leak resistance: the crew roster is page-only (the prompt names no astronaut and no mission), and
the keystone is a DERIVED figure (age at a specific launch date) rather than a memorizable
infobox constant, so parametric recall alone cannot produce it. A linear ReAct agent can solve
this perfectly well — it just has to serialize the seven independent subtasks after the prefix.
"""

from typing import Dict, Any, List, Tuple
import re

from agent.app.idea_test_utils import extract_final_text


LAUNCH_DATE = "December 2, 1993"

# The seven fan-out branches. ``name_rx`` matches the crew member, ``age`` is the derived answer
# on LAUNCH_DATE, ``slug_rx`` the astronaut's own page. Single source of truth for the statement,
# the validators and the compiled plan, so they cannot drift apart.
BRANCHES: List[Dict[str, Any]] = [
    {"astronaut": "Story Musgrave", "age": 58, "dob": "1935-08-19",
     "name_rx": r"musgrave", "slug_rx": r"wiki/story_musgrave"},
    {"astronaut": "Claude Nicollier", "age": 49, "dob": "1944-09-02",
     "name_rx": r"nicollier", "slug_rx": r"wiki/claude_nicollier"},
    {"astronaut": "Jeffrey A. Hoffman", "age": 49, "dob": "1944-11-02",
     "name_rx": r"hoffman", "slug_rx": r"wiki/jeffrey_a\.?_?hoffman"},
    {"astronaut": "Richard O. Covey", "age": 47, "dob": "1946-08-01",
     "name_rx": r"covey", "slug_rx": r"wiki/richard_o\.?_?covey"},
    {"astronaut": "Thomas D. Akers", "age": 42, "dob": "1951-05-20",
     "name_rx": r"akers", "slug_rx": r"wiki/thomas_d\.?_?akers"},
    {"astronaut": "Kathryn C. Thornton", "age": 41, "dob": "1952-08-17",
     "name_rx": r"thornton", "slug_rx": r"wiki/kathryn_(c\.?_?)?thornton"},
    {"astronaut": "Kenneth Bowersox", "age": 37, "dob": "1956-11-14",
     "name_rx": r"bowersox", "slug_rx": r"wiki/ken(neth)?_bowersox"},
]

KEYSTONE = BRANCHES[0]

# Evidence floor: an argmax over a SEVEN-member roster that only ever resolved a handful of
# members is an unjustified guess, so the keystone additionally requires most of the fan-out to
# have landed. The un-gated coverage diagnostic below still reports the exact fraction.
EVIDENCE_FLOOR = 5

# The three prefix waypoints, credited only as a gated intermediate diagnostic.
PREFIX_WAYPOINTS: List[Dict[str, str]] = [
    {"key": "package", "label": "COSTAR", "rx": r"\bcostar\b"},
    {"key": "mission", "label": "STS-61", "rx": r"sts[-\s]?61"},
    {"key": "launch_date", "label": LAUNCH_DATE,
     "rx": r"(2\s+dec|dec\w*\.?\s+2)[^.]{0,12}1993|1993[-/]12[-/]0?2"},
]

# Ages live in 30..70: below it are day-of-month tokens, above it are years. No crew birthday
# falls on a day-of-month inside that band, so a date can never be read as an age.
_AGE_LO, _AGE_HI = 30, 70
_NUM_TOKEN_RX = re.compile(r"\b\d{1,3}\b")
# A sentence-ending period is one NOT followed by a digit, so "58." bounds but "1.5" does not.
_SENT_END_RX = re.compile(r"\.(?!\d)")
_WINDOW = 200

_OLDEST_NEAR_KEYSTONE = re.compile(
    r"(oldest|eldest|most\s+senior|born\s+earliest|earliest[-\s]born)[^.]{0,70}musgrave"
    r"|musgrave[^.]{0,90}(was\s+the\s+oldest|oldest|eldest|most\s+senior|born\s+earliest)",
    re.IGNORECASE,
)
_KEYSTONE_AGE_RX = re.compile(r"\b58\b")


def get_test_metadata() -> Dict[str, Any]:
    """Suite metadata for the runner.

    Returns:
        Dict with test_id, test_name, difficulty_level, category, level and weight.
    """
    return {
        "test_id": "162",
        "test_name": "Tier 5: sequential prefix -> 7-way crew fan-out (oldest astronaut at launch)",
        "difficulty_level": "9/10",
        "category": "Sequential-prefix Fan-out & Aggregation",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    """The mandate. Names ONE starting entity; leaks neither the mission nor any crew member.

    Returns:
        The natural-language task statement handed to the agent.
    """
    return (
        "You are given NO URLs, and no names beyond the one below.\n\n"
        "The Hubble Space Telescope originally carried an instrument called the High Speed "
        "Photometer. It was pulled out of the telescope to make room for the corrective-optics "
        "package that fixed Hubble's blurred vision.\n\n"
        "Work forward from there:\n"
        "  1. Identify that corrective-optics package.\n"
        "  2. Identify the Space Shuttle mission that carried it up and installed it, and read "
        "that mission's exact launch date.\n"
        "  3. From that mission's page, read off its full flight crew.\n\n"
        "Then handle EACH crew member separately: open that astronaut's own page, read their date "
        "of birth from the page (do not guess from memory), and work out how old they were, in "
        "whole years, on the mission's launch date.\n\n"
        "Finally, report (a) WHICH crew member was the OLDEST at launch and their age in whole "
        "years, and (b) the full roster: astronaut -> date of birth -> age at launch, citing the "
        "exact source URL of every astronaut page you read."
    )


def get_required_deliverables() -> List[str]:
    """Returns: the deliverables list shown to the agent and used by reporting."""
    return [
        "The oldest crew member at launch + their age in whole years (the aggregate answer)",
        "The corrective-optics package, the shuttle mission and its launch date",
        "All seven astronaut -> date of birth -> age-at-launch rows",
        "Source URL per astronaut page",
    ]


def get_success_criteria() -> List[str]:
    """Returns: human-readable success criteria for this task."""
    return [
        "Resolves the prefix chain: instrument -> COSTAR -> STS-61 (launched 2 December 1993)",
        "Visits the seven astronaut pages (seven-way fan-out)",
        "Correctly identifies the oldest crew member at launch (Story Musgrave, 58)",
        "Reports all seven astronaut/age-at-launch pairs",
        "Cites each astronaut's source page",
    ]


def _n_visits(observability: Dict[str, Any] = None) -> int:
    """Returns: the recorded page-visit count (0 when observability is missing)."""
    return int((observability or {}).get("visit", {}).get("count", 0) or 0)


def _sentence_window(text: str, span: Tuple[int, int]) -> Tuple[int, int]:
    """Sentence-clipped +-_WINDOW bounds around ``span``.

    Args:
        text: the haystack.
        span: (start, end) of the entity mention.

    Returns:
        (lo, hi) character bounds, clipped at the nearest sentence-ending period.
    """
    s, e = span
    lo = max((m.end() for m in _SENT_END_RX.finditer(text, 0, s)), default=0)
    lo = max(lo, s - _WINDOW)
    m = _SENT_END_RX.search(text, e)
    hi = min(m.start() if m else len(text), e + _WINDOW)
    return lo, hi


def _ages(text: str, lo: int, hi: int) -> List[Tuple[int, int, int]]:
    """Returns: (value, start, end) for every age-plausible integer token in ``text[lo:hi]``."""
    out: List[Tuple[int, int, int]] = []
    for m in _NUM_TOKEN_RX.finditer(text, lo, hi):
        value = int(m.group(0))
        if _AGE_LO <= value <= _AGE_HI:
            out.append((value, m.start(), m.end()))
    return out


def _paired_age_ok(text: str, branch: Dict[str, Any]) -> bool:
    """True when the age token NEAREST this astronaut's name is that astronaut's OWN age.

    Any mention of the astronaut may satisfy the branch (a report may name someone in prose before
    tabulating them), but a borrowed figure never can: if the nearest age at every mention belongs
    to someone else, the branch scores zero. This is the entity-collision gate.

    Args:
        text: the agent's final answer, lower-cased.
        branch: a BRANCHES entry.

    Returns:
        True if the branch is genuinely resolved in the text.
    """
    for m in re.finditer(branch["name_rx"], text, re.IGNORECASE):
        s, e = m.span()
        line_lo = text.rfind("\n", 0, s) + 1
        nl = text.find("\n", e)
        line_hi = len(text) if nl == -1 else nl
        for lo, hi in ((line_lo, line_hi), _sentence_window(text, (s, e))):
            found = _ages(text, lo, hi)
            if not found:
                continue
            nearest = min(found, key=lambda t: 0 if (t[1] <= s and t[2] >= e)
                          else (s - t[2] if t[2] <= s else t[1] - e))[0]
            if nearest == branch["age"]:
                return True
            break
    return False


def _coverage_hits(result: Dict[str, Any]) -> List[str]:
    """Returns: names of the branches whose own age-at-launch is correctly paired with them."""
    text = extract_final_text(result)
    return [b["astronaut"] for b in BRANCHES if _paired_age_ok(text, b)]


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """Keystone predicate: the argmax answer, GROUNDED.

    Credit requires visit.count > 0 — the value string alone is insufficient, else an ungrounded
    parametric guess would bank the gate. It also requires the keystone branch to be genuinely
    resolved (Musgrave paired with 58), so "the oldest was Musgrave" with a fabricated age fails.
    """
    if _n_visits(observability) <= 0:
        return False
    text = extract_final_text(result)
    if not (_OLDEST_NEAR_KEYSTONE.search(text) and _KEYSTONE_AGE_RX.search(text)
            and _paired_age_ok(text, KEYSTONE)):
        return False
    return len(_coverage_hits(result)) >= EVIDENCE_FLOOR


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Visit floor: the golden path is 2 prefix pages + 7 astronaut pages = 9 reads."""
    n = _n_visits(observability)
    return {"check": "visit_count", "passed": n >= 6, "score": min(1.0, n / 9.0),
            "reason": f"{n} visit(s) (target >=9: 2 prefix hops + 7 crew branches; >=6 to pass)"}


def validate_keystone_oldest(result: Dict[str, Any],
                             observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): Story Musgrave named as the oldest at launch, aged 58."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_oldest_at_launch", "passed": passed,
            "score": 1.0 if passed else 0.0,
            "reason": "Oldest at launch = Story Musgrave, 58" if passed
                      else "Oldest crew member at launch (Story Musgrave, 58) missing, "
                           "unpaired or ungrounded"}


def validate_crew_coverage(result: Dict[str, Any],
                           observability: Dict[str, Any]) -> Dict[str, Any]:
    """Breadth diagnostic (UN-gated): how many of the seven branches were actually resolved.

    Deliberately not short-circuited on the keystone. It measures whether the agent really fanned
    out across the discovered roster and derived each age, which is the axis that separates a
    structured agent from a linear one even when the final argmax is botched.
    """
    hits = _coverage_hits(result)
    n = len(BRANCHES)
    return {"check": "crew_coverage", "passed": len(hits) == n, "score": len(hits) / n,
            "reason": f"{len(hits)}/{n} astronaut+age-at-launch pairs resolved "
                      f"({', '.join(hits) or 'none'})"}


def validate_prefix_chain(result: Dict[str, Any],
                          observability: Dict[str, Any]) -> Dict[str, Any]:
    """Sequential-prefix intermediates (gated): COSTAR, STS-61 and the launch date."""
    if not _keystone_ok(result, observability):
        return {"check": "prefix_chain", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> prefix waypoints not credited"}
    text = extract_final_text(result)
    hit = [w["label"] for w in PREFIX_WAYPOINTS if re.search(w["rx"], text, re.IGNORECASE)]
    n = len(PREFIX_WAYPOINTS)
    return {"check": "prefix_chain", "passed": len(hit) == n, "score": len(hit) / n,
            "reason": f"{len(hit)}/{n} prefix waypoints reported ({', '.join(hit) or 'none'})"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Per-branch source URLs (gated on the keystone)."""
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = extract_final_text(result).lower()
    cited = sum(1 for b in BRANCHES if re.search(b["slug_rx"], text))
    n = len(BRANCHES)
    return {"check": "citations", "passed": cited >= 4, "score": cited / n,
            "reason": f"{cited}/{n} astronaut pages cited"}


def get_validation_functions() -> List[callable]:
    """Returns: the ordered deterministic validators for this task."""
    return [validate_visits, validate_keystone_oldest, validate_crew_coverage,
            validate_prefix_chain, validate_citations]


def get_llm_validation_function() -> callable:
    """Returns: None — this task is fully deterministic, no LLM judge."""
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored SEQUENTIAL-PREFIX + FAN-OUT scaffold for the ``graph_compiled`` variant.

    Shape: a 3-leaf chain (package -> mission -> roster) whose roster leaf feeds seven sibling
    slot leaves; the siblings depend on nothing but earlier chain leaves, so the whole fan-out
    wave is mutually independent and runs in parallel. Every leaf id is
    keyed on a GIVEN or on a positional slot, never on a discovered name; the instructions
    template the predecessor's finding ({package}, {mission}, {roster}) instead of stating it.
    The plan leaks no package, no mission designation, no astronaut, no birth date, no age and
    not the argmax — the runtime model still performs every read, derivation and comparison.

    Returns:
        Schema-v2 dict: ``{"leaves": [{id, instruction, expect, depends_on}], "aggregation": str}``.
    """
    package_leaf = {
        "id": "package",
        "instruction": (
            "Open the Wikipedia page for the Hubble Space Telescope instrument called the High "
            "Speed Photometer. It was removed from the telescope to make room for a "
            "corrective-optics package. Report WHICH package replaced it and that package's exact "
            "Wikipedia URL. Do not guess from memory; report no other fact."
        ),
        "expect": "The replacement corrective-optics package — source URL",
        "depends_on": [],
    }
    mission_leaf = {
        "id": "mission",
        "instruction": (
            "Open the Wikipedia page of the package identified in the previous step ({package}). "
            "Find WHICH Space Shuttle mission flew it up and installed it, and that mission's "
            "exact LAUNCH DATE. Report the mission designation, the launch date and the mission's "
            "Wikipedia URL. Do not guess from memory."
        ),
        "expect": "The installing Space Shuttle mission + its exact launch date — source URL",
        "depends_on": ["package"],
    }
    roster_leaf = {
        "id": "roster",
        "instruction": (
            "Open the Wikipedia page of the mission identified in the previous step ({mission}). "
            "Read off its FULL FLIGHT CREW exactly as listed on that page, in the order given, "
            "with each person's role. Report the roster and the source URL. Do not guess from "
            "memory; report no birth dates or ages here."
        ),
        "expect": "The mission's complete flight-crew roster, in page order — source URL",
        "depends_on": ["mission"],
    }
    crew_leaves = [
        {
            "id": f"crew_slot_{i}",
            "instruction": (
                f"Take crew member number {i} from the roster listed in the previous step "
                "({roster}). Open THAT astronaut's own Wikipedia page, read their DATE OF BIRTH "
                "from the page, and then work out how old they were, in whole years, on the "
                "mission launch date established earlier ({mission}). Report the astronaut's name, "
                "their date of birth, the derived age in whole years, and the astronaut page's "
                "exact URL. Do not guess from memory; report nothing about any other crew member."
            ),
            "expect": "ASTRONAUT NAME — date of birth — age in whole years at launch — source URL",
            "depends_on": ["roster", "mission"],
        }
        for i in range(1, len(BRANCHES) + 1)
    ]
    return {
        "leaves": [package_leaf, mission_leaf, roster_leaf] + crew_leaves,
        "aggregation": (
            "You now have the corrective-optics package, the shuttle mission that installed it "
            "with its launch date, the mission's full crew roster, and for every crew member a "
            "date of birth plus an age in whole years at launch. AGGREGATE across the WHOLE crew: "
            "determine who had the MAXIMUM age at launch. Report (a) that crew member and their "
            "age in whole years, stating explicitly that they were the oldest at launch, and (b) "
            "the full roster astronaut -> date of birth -> age at launch, citing every astronaut "
            "page URL. If any crew member's age is missing, say so rather than inventing it."
        ),
    }
