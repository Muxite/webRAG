"""
Test 158: Tier 5 (breadth) — GENUINE WIDE FAN-OUT, 7 fully independent airport lookups
framed as a natural charter-planning question ("is there an island on our shortlist we have
to drop?").
Level: graph   Weight: long   Difficulty: 8/10

NO URLs are given. SEVEN Greek islands are named. For EACH island the agent must (1) work out
which airport actually serves that island, (2) open that airport's own page, and (3) read the
RUNWAY LENGTH in metres out of the infobox runway table. Nothing learned about one island is
needed for any other island — the seven arms are genuinely independent, one to three tool
calls each (search -> maybe one disambiguation -> visit). The difficulty is BREADTH, not depth.

WHY IT IS SHAPED THIS WAY (mechanism-suite role, see DAG_V3_LEDGER_MASTER_PLAN §7/§8.3):
  * This is the suite's clean wide-fan-out HOLDOUT. The earlier breadth pilot found the DAG v2
    graph engine stalls at 0-4/7 visits on genuine fan-out; this task is a well-formed instance
    of exactly that shape, so any architecture change (Phase-0 ablation arms, or a future
    evidence-queue engine) has to earn its breadth claim here.
  * It is deliberately NOT a candidate x field slot-filling table: the user asks a single
    ordinary question (can we fly our chartered turboprop to every island on our shortlist?),
    the runway figures are working notes, and the answer is an ISLAND NAME, not a table cell.
    A deterministic queue that overfits to "enumerate entities, fill one column" gains nothing
    structural here beyond actually doing all seven lookups.
  * The keystone is an EXCLUSIVE claim ("this is the only island we must drop"), which asserts
    something about all seven islands. It therefore cannot be honestly answered from one lookup,
    which is why the keystone additionally requires a majority of the seven runway figures to
    have been gathered.

Ground truth — every runway figure verified against the live English Wikipedia article infobox
(``r1-length-m`` / ``r1-length-f`` in the raw wikitext) on 2026-08-25:

  island      article title                                              IATA  runway    >= 1,300 m?  margin
  ------------------------------------------------------------------------------------------------------
  Naxos       Naxos Island National Airport                              JNX     901 m      NO         -399 m
  Skiathos    Skiathos International Airport "Alexandros Papadiamantis"  JSI   1,628 m      YES        +328 m
  Mykonos     Mykonos-Manto Mavrogenous Airport                          JMK   1,903 m      YES        +603 m
  Samos       Samos International Airport "Aristarchos of Samos"         SMI   2,100 m      YES        +800 m
  Santorini   Santorini International Airport                            JTR   2,197 m      YES        +897 m
  Kos         Kos International Airport "Ippokratis"                     KGS   2,400 m      YES      +1,100 m
  Rhodes      Rhodes International Airport "Diagoras"                    RHO   3,305 m      YES      +2,005 m
  ------------------------------------------------------------------------------------------------------
  KEYSTONE = the ONLY island whose airport cannot take the aircraft = NAXOS (901 m)

MARGINS: the aircraft minimum is 1,300 m. The failing island misses it by 399 m (31% of the
threshold) and the closest passing island clears it by 328 m (25%). Every infobox figure is an
exact integer stated in both metres and feet on the page, and the two figures bracketing the
threshold are 901 m and 1,628 m — a 727 m EMPTY BAND. No plausible single misread can flip the
keystone, and no misread of one arm can invent a second failing island.

ANTI-PARAMETRIC: the surprising item is the keystone. Naxos is the LARGEST of the Cyclades, so
the "smallest island loses" prior points the wrong way, and 901 m is a page-only figure. The
other six runway lengths (1,628 / 1,903 / 2,100 / 2,197 / 2,400 / 3,305) are not simultaneously
recallable, and the derived claim ("Naxos is the only one below 1,300 m") is published nowhere.

ENTITY-COLLISION SURFACE (deliberate, and what the validators are hardened against):
  * Rhodes has a SECOND airport page, Rhodes Maritsa Airport (military, ex-civil), whose runways
    are 2,400 m and 1,200 m — the 2,400 m figure collides exactly with KOS's real figure, and the
    1,200 m one would fabricate a second failing island.
  * Santorini is also "Thira"; Skiathos/Kos/Samos/Mykonos pages all carry an honorific name, and
    Naxos is one island away from Paros, whose airport figure is a common substitution.
  Coverage therefore credits an island only when its OWN figure is the measurement nearest that
  island's name — a value borrowed from another island scores zero for that arm.

KEYSTONE (hard 0/1)  = Naxos named as the island to drop, NO other island named as droppable,
                       plus grounding (>=1 visit) and >= 5/7 runway figures actually gathered.
COVERAGE (UN-gated)  = how many of the seven (island, own runway length) pairs were gathered.
CLASSIFICATION (un-gated) = how many of the seven per-island verdicts (usable / must drop) were
                       stated correctly — the breadth axis that survives a botched final answer.
Secondary (GATED on keystone) = the 901 m figure, the six usable islands, source URLs.
"""

from typing import Dict, Any, List, Optional, Tuple
import re
from agent.app.idea_test_utils import extract_final_text


# --- verified fixtures -------------------------------------------------------------------
# 'runway_m' = the infobox r1-length-m figure (live-verified 2026-08-25).
# 'viable'   = (runway_m >= MIN_RUNWAY_M).  Nothing below is leaked into the task statement or
# the compiled plan: the statement names only the ISLANDS and the GIVEN aircraft minimum.

MIN_RUNWAY_M: int = 1300  # metres of paved runway the chartered aircraft needs (a GIVEN)

ENTITIES: List[Dict[str, Any]] = [
    {
        "key":      "naxos",
        "island":   "Naxos",
        "airport":  "Naxos Island National Airport",
        "runway_m": 901,
        "viable":   False,
        "name_rx":  r"\bnaxos\b",
        # Accepted measurement tokens for this arm: the metres figure, a rounded metres reading,
        # and the page's own feet figure. Anything else nearest the island name is a mis-pairing.
        "values":   (901, 900, 2957),
        "slug_rx":  r"wiki/naxos",
    },
    {
        "key":      "skiathos",
        "island":   "Skiathos",
        "airport":  'Skiathos International Airport "Alexandros Papadiamantis"',
        "runway_m": 1628,
        "viable":   True,
        "name_rx":  r"\bskiathos\b",
        "values":   (1628, 1630, 5341),
        "slug_rx":  r"wiki/skiathos",
    },
    {
        "key":      "mykonos",
        "island":   "Mykonos",
        "airport":  "Mykonos-Manto Mavrogenous Airport",
        "runway_m": 1903,
        "viable":   True,
        "name_rx":  r"\bmykonos\b",
        "values":   (1903, 1900, 6244),
        "slug_rx":  r"wiki/mykonos",
    },
    {
        "key":      "samos",
        "island":   "Samos",
        "airport":  'Samos International Airport "Aristarchos of Samos"',
        "runway_m": 2100,
        "viable":   True,
        "name_rx":  r"\bsamos\b",
        "values":   (2100, 6890),
        "slug_rx":  r"wiki/samos",
    },
    {
        "key":      "santorini",
        "island":   "Santorini",
        "airport":  "Santorini International Airport",
        "runway_m": 2197,
        "viable":   True,
        # The island is also called Thira; the airport page uses both.
        "name_rx":  r"\bsantorini\b|\bthira\b",
        "values":   (2197, 2200, 7208),
        "slug_rx":  r"wiki/santorini|wiki/thira",
    },
    {
        "key":      "kos",
        "island":   "Kos",
        "airport":  'Kos International Airport "Ippokratis"',
        "runway_m": 2400,
        "viable":   True,
        "name_rx":  r"\bkos\b",
        "values":   (2400, 7874),
        "slug_rx":  r"wiki/kos[_a-z]*",
    },
    {
        "key":      "rhodes",
        "island":   "Rhodes",
        "airport":  'Rhodes International Airport "Diagoras"',
        "runway_m": 3305,
        "viable":   True,
        "name_rx":  r"\brhodes\b|\brodos\b",
        "values":   (3305, 3300, 10844),
        "slug_rx":  r"wiki/rhodes",
    },
]

VIABLE: List[Dict[str, Any]] = [e for e in ENTITIES if e["viable"]]          # 6 islands
DROPPED: List[Dict[str, Any]] = [e for e in ENTITIES if not e["viable"]]     # 1 island (Naxos)
KEYSTONE_ISLAND: str = DROPPED[0]["island"]                                  # "Naxos"
KEYSTONE_RUNWAY_M: int = DROPPED[0]["runway_m"]                              # 901
# Minimum number of (island, runway) pairs the keystone claim has to rest on. The keystone is an
# EXCLUSIVE claim about all seven islands, so a run that looked up one island and guessed must
# not bank it; a majority-plus (5/7) is the evidentiary floor.
KEYSTONE_MIN_COVERAGE: int = 5

# Measurement tokens are plain numbers, optionally thousands-grouped, optionally decimal.
_NUM_TOKEN_RX = re.compile(r"\d[\d,]*(?:\.\d+)?")
# A runway measurement is between these bounds in either unit; this drops runway designations
# ("18/36" -> 18, 36), elevations ("10 ft") and passenger counts ("86,210") without ever
# touching a real length figure (real range here: 901 m .. 10,844 ft).
_MEASURE_LO, _MEASURE_HI = 300, 20000


# --- metadata ----------------------------------------------------------------------------

def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id":          "158",
        "test_name": (
            "Tier 5: Genuine 7-way fan-out — which shortlisted Greek island has too short a "
            "runway for the chartered aircraft"
        ),
        "difficulty_level": "8/10",
        "category":         "Breadth Fan-out & Eligibility Fit",
        "level":            "graph",
        "weight":           "long",
    }


# --- task statement ----------------------------------------------------------------------

def get_task_statement() -> str:
    listing = ", ".join(e["island"] for e in ENTITIES)
    return (
        "We are chartering one small turboprop for a two-week island-hopping trip around "
        "Greece next spring instead of piecing the route together from ferries. The charter "
        "operator has told us the aircraft needs at least "
        f"{MIN_RUNWAY_M:,} metres of paved runway at our planned take-off weights.\n\n"
        f"Our shortlist of islands is: {listing}.\n\n"
        "Before we pay the deposit we need to know whether any island on that shortlist has to "
        "be dropped because the airport serving it has too short a runway for this aircraft. "
        "Work each island out separately — which airport actually serves it, and how long that "
        "airport's runway is. Nothing you find out about one island affects any other one.\n\n"
        "You are given NO URLs: search for the pages you need and READ them. Take the runway "
        "length from the airport's own page (the runway table / infobox runway length in "
        "metres) — not from a search snippet and not from memory. Be careful that you have the "
        "right airport: some islands have more than one airfield, and a nearby island's airport "
        "is not a substitute.\n\n"
        "Please report:\n"
        f"  (a) which island(s), if any, we must drop from the shortlist because the runway is "
        f"under {MIN_RUNWAY_M:,} m — and confirm the rest are fine. Put this in your primary "
        "answer; if exactly one island fails, name that island.\n"
        "  (b) the runway length in metres for each of the seven islands, and the name of the "
        "airport you read it from,\n"
        "  (c) for each island, whether it clears the "
        f"{MIN_RUNWAY_M:,} m minimum, and\n"
        "  (d) the exact source URL of every airport page you read."
    )


# --- deliverables / success criteria -----------------------------------------------------

def get_required_deliverables() -> List[str]:
    return [
        f"The island(s) that must be dropped because the runway is under {MIN_RUNWAY_M:,} m "
        "(the keystone answer)",
        "Runway length in metres for each of the seven islands, with the airport it was read from",
        f"Per-island verdict: clears / does not clear the {MIN_RUNWAY_M:,} m minimum",
        "Source URL for each airport page read",
    ]


def get_success_criteria() -> List[str]:
    viable_names = ", ".join(e["island"] for e in VIABLE)
    return [
        f"At least {len(ENTITIES) - 1} pages visited (target {len(ENTITIES)}: one airport per island)",
        f"Correctly names {KEYSTONE_ISLAND} as the only island that must be dropped "
        f"(runway {KEYSTONE_RUNWAY_M} m, under the {MIN_RUNWAY_M:,} m minimum)",
        "Names no other island as unusable",
        f"Reports the runway length of all seven islands' airports, each paired with the right island",
        f"Confirms the six usable islands: {viable_names}",
        "Cites a source URL for each airport page",
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


def _n_visits(observability: Dict[str, Any] = None) -> int:
    return int((observability or {}).get("visit", {}).get("count", 0) or 0)


# --- measurement pairing (entity-collision hardened) --------------------------------------

# Window (chars) searched around an island name when its own line carries no measurement.
# Newlines are tolerated on purpose: 'Naxos\n  Airport: ...\n  Runway: 901 m' is a normal
# report layout. The window is clipped at the nearest SENTENCE-ending period so one island's
# figure cannot bleed in from a neighbouring sentence.
_WINDOW = 220
# A sentence-ending period is one NOT followed by a digit, so '2,197.0' / '1.5' never look like
# a sentence boundary.
_SENT_END_RX = re.compile(r"\.(?!\d)")


def _measurements(text: str, lo: int, hi: int) -> List[Tuple[int, int, int]]:
    """(value, start, end) for every plausible runway measurement in ``text[lo:hi]``.

    The GIVEN minimum (1,300 m) is dropped — it is restated on almost every row and is never
    the island's own figure — as are tokens outside the physical range of a runway length.
    """
    out: List[Tuple[int, int, int]] = []
    for m in _NUM_TOKEN_RX.finditer(text, lo, hi):
        raw = m.group(0).replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        if value != int(value):
            continue
        ivalue = int(value)
        if ivalue == MIN_RUNWAY_M or not (_MEASURE_LO <= ivalue <= _MEASURE_HI):
            continue
        out.append((ivalue, m.start(), m.end()))
    return out


def _sentence_window(text: str, span: Tuple[int, int]) -> Tuple[int, int]:
    """Sentence-clipped +-_WINDOW bounds around ``span``."""
    s, e = span
    lo = max((m.end() for m in _SENT_END_RX.finditer(text, 0, s)), default=0)
    lo = max(lo, s - _WINDOW)
    m = _SENT_END_RX.search(text, e)
    hi = min(m.start() if m else len(text), e + _WINDOW)
    return lo, hi


def _paired_value_ok(text: str, entity: Dict[str, Any]) -> bool:
    """True when the measurement NEAREST this island's name is this island's OWN runway figure.

    Any occurrence of the island name may satisfy this (a report may mention an island in prose
    before tabulating it), but a borrowed figure never can: if the nearest measurement to every
    occurrence belongs to a different island (or to nothing at all), the arm scores zero. This
    is the entity-collision gate — 'Samos: 2,197 m' (Santorini's figure) is not coverage.
    """
    for m in re.finditer(entity["name_rx"], text, re.IGNORECASE):
        s, e = m.span()
        line_lo = text.rfind("\n", 0, s) + 1
        nl = text.find("\n", e)
        line_hi = len(text) if nl == -1 else nl
        for lo, hi in ((line_lo, line_hi), _sentence_window(text, (s, e))):
            found = _measurements(text, lo, hi)
            if not found:
                continue
            nearest = min(found, key=lambda t: 0 if (t[1] <= s and t[2] >= e)
                          else (s - t[2] if t[2] <= s else t[1] - e))[0]
            if nearest in entity["values"]:
                return True
            break  # this occurrence pairs with some OTHER value -> it does not credit the arm
    return False


def _coverage_hits(text: str) -> List[str]:
    """Islands whose OWN runway figure is correctly paired with their name."""
    return [e["island"] for e in ENTITIES if _paired_value_ok(text, e)]


def _credited_coverage(result: Dict[str, Any], observability: Dict[str, Any] = None) -> int:
    """Coverage capped by visit count: a zero-visit recall answer banks no breadth credit."""
    return min(len(_coverage_hits(_all_text(result))), _n_visits(observability))


# --- per-island verdicts ------------------------------------------------------------------

# Only TRUE verdict triggers. Bare comparison glyphs ('>' / '<') are excluded on purpose: a
# coverage row 'Naxos -> 901 m' must not be read as a verdict about anything.
_VIABLE_CUE = (
    r"yes\b|✓|✔|\btrue\b|\bcan(?!not)\b(?!['’]t)|\bable\b|viable\b|suitable\b|usable\b"
    r"|keep\b|long enough\b|meets?\b|exceed(?:s|ing|ed)?\b|above\b|\bover\b|longer\b"
    r"|at least\b|sufficient\b|works?\b|eligible\b|pass(?:es|ed)?\b|clears?\b|fine\b|\bok\b"
)
_EXCLUDE_CUE = (
    r"\bno\b|✗|✘|\bfalse\b|too short\b|drop(?:s|ped|ping)?\b|exclude[sd]?\b|remove[sd]?\b"
    r"|cannot\b|can['’]t|unable\b|ineligible\b|unsuitable\b|insufficient\b|below\b|under\b"
    r"|shorter\b|less than\b|except\b|fail(?:s|ed)?\b|skip\b"
)
_VIABLE_RX = re.compile(_VIABLE_CUE, re.IGNORECASE)
_EXCLUDE_RX = re.compile(_EXCLUDE_CUE, re.IGNORECASE)
# A negation immediately in front of a cue flips its polarity: 'not exceeding 1,300 m' is an
# exclusion, and 'not under 1,300 m' (the shape the deterministic composer renders) is not.
_NEGATION_RX = re.compile(r"(?:\bnot|n['’]t|\bno)\s*$", re.IGNORECASE)


def _cue_polarity(text: str, start: int, is_viable: bool) -> bool:
    return (not is_viable) if _NEGATION_RX.search(text[max(0, start - 12):start]) else is_viable


def _direction_in(text: str, span: Tuple[int, int], lo: int, hi: int) -> Optional[bool]:
    """Polarity of the verdict cue NEAREST ``span`` within ``text[lo:hi]``, or None if none."""
    s, e = span
    best_dist: Optional[int] = None
    best: Optional[bool] = None
    for rx, is_viable in ((_VIABLE_RX, True), (_EXCLUDE_RX, False)):
        for m in rx.finditer(text, lo, hi):
            if m.start() >= s and m.end() <= e:
                continue  # a cue inside the island name itself is not a verdict
            dist = (s - m.end()) if m.end() <= s else (m.start() - e)
            if dist < 0:
                continue
            if best_dist is None or dist < best_dist:
                best_dist, best = dist, _cue_polarity(text, m.start(), is_viable)
    return best


def _verdict(text: str, entity: Dict[str, Any]) -> Optional[bool]:
    """The verdict this report attaches to one island, or None when it states none.

    The island's OWN LINE wins first — in a two-list layout ('Usable: A, B, C' / 'Drop: D') the
    last name on a line sits closer to the NEXT line's heading than to its own, so a pure
    character-distance rule would flip it. The FIRST occurrence carrying any cue decides, so a
    later restatement cannot rescue a wrong verdict.
    """
    for m in re.finditer(entity["name_rx"], text, re.IGNORECASE):
        s, e = m.span()
        line_lo = text.rfind("\n", 0, s) + 1
        nl = text.find("\n", e)
        line_hi = len(text) if nl == -1 else nl
        same_line = _direction_in(text, (s, e), line_lo, line_hi)
        if same_line is not None:
            return same_line
        lo, hi = _sentence_window(text, (s, e))
        widened = _direction_in(text, (s, e), lo, hi)
        if widened is not None:
            return widened
    return None


def _excluded_set(text: str) -> set:
    """Islands this report says must be dropped."""
    return {e["island"] for e in ENTITIES if _verdict(text, e) is False}


# --- keystone -----------------------------------------------------------------------------

def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE gate, three conditions, all required:

      1. GROUNDING: at least one page was actually visited.
      2. EVIDENCE FLOOR: at least KEYSTONE_MIN_COVERAGE of the seven (island, own runway) pairs
         were gathered — the answer is an exclusive claim over all seven islands, so a single
         lookup plus a guess must not score.
      3. THE ANSWER: the reported set of islands to drop is EXACTLY {Naxos}. Naming a second
         island (the Rhodes-Maritsa 1,200 m trap) fails, as does naming none or the wrong one.
    """
    if _n_visits(observability) <= 0:
        return False
    if _credited_coverage(result, observability) < KEYSTONE_MIN_COVERAGE:
        return False
    excluded = _excluded_set(_primary_text(result))
    if not excluded:
        # Terse primary slots ('Naxos') carry no verdict cue; fall back to the whole report,
        # where the same exclusivity rule then has to hold.
        excluded = _excluded_set(_all_text(result))
    return excluded == {KEYSTONE_ISLAND}


# --- validation functions ------------------------------------------------------------------

def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a seven-way independent fan-out wants one airport page per island."""
    n = _n_visits(observability)
    need = len(ENTITIES) - 1  # >= 6
    return {
        "check":  "visit_count",
        "passed": n >= need,
        "score":  min(1.0, n / len(ENTITIES)),
        "reason": (
            f"{n} visit(s) (target >= {len(ENTITIES)}: one airport page per island; "
            f">= {need} to pass)"
        ),
    }


def validate_keystone_dropped_island(result: Dict[str, Any],
                                     observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): Naxos, and only Naxos, named as the island to drop — grounded, and
    resting on at least 5/7 gathered runway figures."""
    passed = _keystone_ok(result, observability)
    return {
        "check":  "keystone_dropped_island",
        "passed": passed,
        "score":  1.0 if passed else 0.0,
        "reason": (
            f"{KEYSTONE_ISLAND} correctly identified as the only island to drop, grounded and "
            f"backed by >= {KEYSTONE_MIN_COVERAGE}/{len(ENTITIES)} gathered runway figures"
            if passed else
            f"Did not establish {KEYSTONE_ISLAND} ({KEYSTONE_RUNWAY_M} m) as the ONLY island "
            f"below the {MIN_RUNWAY_M:,} m minimum "
            f"(reported: {sorted(_excluded_set(_all_text(result))) or 'none'}; coverage "
            f"{_credited_coverage(result, observability)}/{len(ENTITIES)}, "
            f"{_n_visits(observability)} visit(s))"
        ),
    }


def validate_runway_coverage(result: Dict[str, Any],
                             observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the seven (island, own runway length) pairs were
    gathered. Deliberately NOT short-circuited on the keystone — this is the axis that separates
    a structured agent that fanned out to all seven airports from a linear one that ran out of
    budget, even when the final answer is botched. Capped by visit count."""
    text = _all_text(result)
    hits = _coverage_hits(text)
    credited = min(len(hits), _n_visits(observability))
    n = len(ENTITIES)
    return {
        "check":  "runway_coverage",
        "passed": credited == n,
        "score":  credited / n,
        "reason": (
            f"{credited}/{n} (island, runway length) pairs gathered "
            f"({', '.join(hits[:credited]) if credited else 'none'}; {len(hits)} correctly "
            f"paired, {_n_visits(observability)} visit(s))"
        ),
    }


def validate_island_verdicts(result: Dict[str, Any],
                             observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated per-island diagnostic: how many of the seven usable / must-drop verdicts were
    stated correctly, independent of the final answer. Capped by visit count."""
    text = _all_text(result)
    hits = [e["island"] for e in ENTITIES if _verdict(text, e) is e["viable"]]
    credited = min(len(hits), _n_visits(observability))
    n = len(ENTITIES)
    return {
        "check":  "island_verdicts",
        "passed": credited == n,
        "score":  credited / n,
        "reason": (
            f"{credited}/{n} per-island verdicts stated correctly "
            f"({', '.join(hits[:credited]) if credited else 'none'})"
        ),
    }


def validate_dropped_runway_value(result: Dict[str, Any],
                                  observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the failing island's actual runway figure (901 m), paired with its name.
    Short-circuits to 0 without the keystone, so a wrong answer banks nothing here."""
    if not _keystone_ok(result, observability):
        return {
            "check":  "dropped_runway_value",
            "passed": False,
            "score":  0.0,
            "reason": "Keystone absent -> runway figure of the dropped island not credited",
        }
    ok = _paired_value_ok(_all_text(result), DROPPED[0])
    return {
        "check":  "dropped_runway_value",
        "passed": ok,
        "score":  1.0 if ok else 0.0,
        "reason": (
            f"{KEYSTONE_ISLAND}'s runway ({KEYSTONE_RUNWAY_M} m) reported"
            if ok else
            f"{KEYSTONE_ISLAND}'s own runway figure ({KEYSTONE_RUNWAY_M} m) not reported"
        ),
    }


def validate_viable_islands(result: Dict[str, Any],
                            observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the six usable islands are named. Short-circuits to 0 without the keystone."""
    if not _keystone_ok(result, observability):
        return {
            "check":  "viable_islands",
            "passed": False,
            "score":  0.0,
            "reason": "Keystone absent -> usable-island list not credited",
        }
    text = _all_text(result)
    hits = [e["island"] for e in VIABLE if re.search(e["name_rx"], text, re.IGNORECASE)]
    n = len(VIABLE)
    return {
        "check":  "viable_islands",
        "passed": len(hits) == n,
        "score":  len(hits) / n,
        "reason": f"{len(hits)}/{n} usable islands named ({', '.join(hits) or 'none'})",
    }


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: one source URL per airport page. Short-circuits to 0 without the keystone."""
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
        "reason": f"{cited}/{n} airport pages cited (>= 5 to pass)",
    }


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_dropped_island,
        validate_runway_coverage,
        validate_island_verdicts,
        validate_dropped_runway_value,
        validate_viable_islands,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge.
    return None


# --- compiled plan -----------------------------------------------------------------------

def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    SEVEN leaves, EVERY ONE with an empty ``depends_on``: the arms are genuinely independent, so
    the whole plan is one parallel wave plus an aggregation. Each leaf ``id`` is keyed on the
    ISLAND (a GIVEN from the mandate), never on an airport name, a runway figure or the answer.
    Airport identification stays INSIDE the leaf — the plan does not name a single airport.

    Encodes STRUCTURE and the GIVEN minimum only: no runway length, no per-island verdict and no
    keystone island is leaked anywhere in the plan text.
    """
    leaves = [
        {
            "id": f"{e['key']}_runway",
            "instruction": (
                f"Find the airport that serves the Greek island of {e['island']} (its own "
                "commercial airport, not an airport on a neighbouring island), open that "
                "airport's own Wikipedia article, and read the LENGTH OF ITS RUNWAY IN METRES "
                "straight from the runway table in the infobox. If the article lists more than "
                "one runway, report the longest. Report ONLY the airport's name, that single "
                "runway length in metres, and the exact source URL. Do not report the runway "
                "width, the elevation, passenger numbers, or a figure taken from memory."
            ),
            "expect": (
                f"AIRPORT serving {e['island']} and its RUNWAY LENGTH in metres (a single "
                "number) -- source URL"
            ),
            "depends_on": [],
        }
        for e in ENTITIES
    ]
    restate = "\n".join(f"  {e['island']} -> [airport], [runway] m" for e in ENTITIES)
    return {
        "leaves": leaves,
        # Deterministic composition: the executor applies the minimum in Python over the seven
        # gathered runway lengths (zero extra LLM calls). Free-text threshold comparison across
        # seven rows is the confirmed failure mode even when every figure is right. If any leaf
        # fails to resolve, the composer returns nothing and the recipe below runs unchanged.
        "agg_mode": "computed",
        "composition": {
            "op": "count_threshold",
            "answer_noun": "island",
            "value_label": "runway length",
            "unit": "m",
            "comparator": ">=",
            "threshold": MIN_RUNWAY_M,
            "items": [
                {"leaf": f"{e['key']}_runway", "label": e["island"], "type": "number"}
                for e in ENTITIES
            ],
        },
        "aggregation": (
            "You now have, for each of the seven shortlisted islands, the airport that serves it "
            "and that airport's runway length in metres, with a source URL. Before judging "
            "anything, RESTATE each island's airport and runway length explicitly in this "
            "format:\n"
            f"{restate}\n"
            "(substituting what you retrieved for each '[airport]' and '[runway]' placeholder).\n\n"
            "Then, FOR EACH ISLAND IN TURN, state whether its runway is at least "
            f"{MIN_RUNWAY_M:,} metres long. Finally answer the traveller's question:\n"
            "  (a) name the island(s), if any, that must be dropped from the shortlist because "
            f"the runway is under {MIN_RUNWAY_M:,} m — this is the primary answer; if exactly "
            "one island fails, name that island and say the other six are fine,\n"
            "  (b) list all seven runway lengths as you restated them above,\n"
            "  (c) give each island's clears / does not clear verdict, and\n"
            "  (d) give the source URL for each airport page."
        ),
    }
