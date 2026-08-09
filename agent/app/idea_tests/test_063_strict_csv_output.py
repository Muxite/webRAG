"""
Test 063: Strict structured output under research load — CSV contract (rule-adherence axis)
Level: integration   Weight: long   Difficulty: 7/10

The CSV sibling of the proven strict-JSON task (057). Same thesis, broadened format axis: the
chains/fan-outs score *what was found*; this one scores *whether the rigid output contract was
honoured while finding it*. The agent must research 4 obscure chemical elements x 2 page-readable
fields and emit the result as ONLY a strict 4-row CSV with an EXACT header line, EXACT column
order, no prose, no markdown fence, no extra/missing rows. This is the suite's CSV rule-adherence
discriminator, and the one maximally favouring the compiled scaffold: a cheap model that must
*reason and format at once* leaks prose, fences the block, or mangles the header; the compiled
plan's leaves fill atomic cells while the AGGREGATION step emits the rigid CSV deterministically
— the harness owns the format.

Bimodal-by-design:
  * KEYSTONE (0/1): the deliverable is a BYTE-EXACT CSV — its first non-empty line is the EXACT
    header, it carries EXACTLY 4 three-column data rows, AND the keystone element's keystone field
    is exactly correct. Unlike 057's JSON keystone (which tolerates a benign ```json fence), the
    CSV keystone does NOT strip a fence and does NOT tolerate a prose preamble: a fence pushes
    "```csv" into the first line and a preamble pushes prose there, so either one fails the
    first-line-is-header test -> keystone 0. (Format failure tanks the keystone; right format AND
    the right looked-up value are fused into the one gate.)
  * COVERAGE (UN-gated, text-based regex over the raw answer): how many of the 4 elements x both
    fields were actually gathered — credited even when the CSV is botched, so a linear agent that
    found the facts but mangled the format still registers its gathering. This is the axis that
    separates a structured agent from a linear one when the final answer is botched.
  * SCHEMA (all-values) + STRICT-FORMAT (secondaries): short-circuit to 0 when the keystone is
    absent — exactly-4-rows with the EXACT atomic-number AND kelvin value per element, and "ONLY
    the CSV, no prose/fence/extra-whitespace".

Boundary chosen (and unit-tested): the KEYSTONE is value-TOLERANT about benign cell whitespace
("Erbium, 68, 1802" still parses to the right ints) so a single spacing slip does not nuke the
gate; the STRICT-FORMAT secondary is byte-strict (no spaces around commas, exactly header+4 rows,
nothing else), so a spaced-but-correct CSV keeps the keystone (1.0) yet fails strict-format (0.0).
A ```csv fence or a prose preamble breaks BOTH (the keystone's first-line check rejects them).

Parametric-leak hardening: the two fields per element are the ATOMIC NUMBER (memorable / parametric
for a capable model) and the MELTING POINT in WHOLE KELVIN (page-only). The keystone is pinned on
the least-memorable, least-round metric in the set — Erbium's kelvin melting point (1802) — so a
from-memory attempt misses the keystone value. Chemistry sources routinely cite melting points in
°C (1529 °C for Erbium), not as K infobox integers, and Erbium has no strong pop-culture
associations; the Celsius equivalent (1529 °C) differs from the Kelvin value (1802 K) by exactly
273, making any K/°C confusion immediately detectable. The compiled arm reads the K value off the
infobox with an unambiguous extraction instruction -> 1; the parametric arm has no reliable path to
the exact integer 1802 -> 0.

Ground truth (verified against live English Wikipedia element infoboxes, 2026-06-27):
  Samarium  (Sm) -> atomic number 62 -> melting point 1345 K (1072 °C, 1962 °F)
      https://en.wikipedia.org/wiki/Samarium
  Gadolinium (Gd) -> atomic number 64 -> melting point 1585 K (1312 °C, 2394 °F)
      https://en.wikipedia.org/wiki/Gadolinium
  Erbium    (Er) -> atomic number 68 -> melting point 1802 K (1529 °C, 2784 °F)   [KEYSTONE]
      https://en.wikipedia.org/wiki/Erbium
  Lutetium  (Lu) -> atomic number 71 -> melting point 1925 K (1652 °C, 3006 °F)
      https://en.wikipedia.org/wiki/Lutetium
The keystone (Erbium melting_point_k == 1802) is the least-guessable value in the set: Erbium is
an obscure heavy lanthanide without strong pop-culture associations; chemistry databases quote its
melting point as 1529 °C (not 1802 K), so a parametric model that recalls the Celsius value still
fails the Kelvin-gate; the value 1802 is non-round (not a multiple of 5, 10, 50, or 100). The
only other temperature on Erbium's page is the boiling point (3141 K), which differs from every
fixture number, so it neither flips the keystone nor false-triggers coverage.
All eight values are mutually distinct (melting points 1345/1585/1802/1925; atomic numbers
62/64/68/71; no melting point equals an atomic number), so the \\b-anchored coverage regexes never
collide. The Celsius values (1072/1312/1529/1652) are also distinct from every K value in the set.
"""

from typing import Dict, Any, List, Optional
import csv
import io
import re
from agent.app.idea_test_utils import extract_final_text


# Single source of truth for the task statement, validators and the compiled plan, so they can
# never drift. ``name_rx`` matches a reported element name (case-insensitive); the integers are the
# verified infobox figures; ``slug`` is the page slug (for documentation of provenance).
ENTRIES: List[Dict[str, Any]] = [
    {"name": "Samarium", "name_rx": r"samarium",
     "atomic_number": 62, "melting_point_k": 1345, "slug": r"wiki/Samarium"},
    {"name": "Gadolinium", "name_rx": r"gadolinium",
     "atomic_number": 64, "melting_point_k": 1585, "slug": r"wiki/Gadolinium"},
    # KEYSTONE element. Erbium's Kelvin melting point (1802) is the least-guessable value in the
    # set: chemistry sources cite 1529 °C, not the K integer, so parametric recall fails the gate.
    {"name": "Erbium", "name_rx": r"erbium",
     "atomic_number": 68, "melting_point_k": 1802, "slug": r"wiki/Erbium"},
    {"name": "Lutetium", "name_rx": r"lutetium",
     "atomic_number": 71, "melting_point_k": 1925, "slug": r"wiki/Lutetium"},
]

# The strict output contract.
HEADER = "element,atomic_number,melting_point_k"
HEADER_COLS = ("element", "atomic_number", "melting_point_k")

# Keystone: byte-exact CSV format AND the Erbium row's melting point (page-only, non-round).
KEYSTONE_NAME_RX = r"erbium"
KEYSTONE_FIELD = "melting_point_k"
KEYSTONE_VALUE = 1802

# A byte-strict data row: a single-word element name, then two bare integers, no spaces.
_STRICT_ROW_RE = re.compile(r"^[A-Za-z]+,\d+,\d+$")


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "063",
        "test_name": "Strict CSV structured output under research load (rule adherence)",
        "difficulty_level": "7/10",
        "category": "Strict structured output (rule adherence)",
        "level": "integration",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. '{e['name']}'" for i, e in enumerate(ENTRIES, 1))
    return (
        "You are given NO URLs. For EACH of the following four chemical elements, search for its "
        "Wikipedia page and read two facts DIRECTLY from the page (do not guess from memory): "
        "the ATOMIC NUMBER, and the MELTING POINT in WHOLE KELVIN (the integer before 'K' in the "
        "infobox 'Melting point' row):\n"
        f"{listing}\n\n"
        "Return your answer as a STRICT CSV with EXACTLY five lines: one header line followed by "
        "EXACTLY four data rows, one per element, in the SAME column order as the header. The "
        "header line must be EXACTLY (character for character):\n"
        f"  {HEADER}\n"
        "Each data row must be: the element name, then its atomic number (an integer), then its "
        "melting point in whole kelvin (an integer), separated by single commas with NO spaces "
        "around the commas, e.g. 'Element,12,3456'.\n\n"
        "Output ONLY the CSV. Do NOT include any prose, explanation, markdown code fences (no "
        "```), trailing commentary, units, thousands separators, a repeated header, or extra / "
        "missing rows. The FIRST line of your answer must be the header above, and the LAST line "
        "must be the fourth data row."
    )


def get_required_deliverables() -> List[str]:
    return [
        "A strict CSV: exactly one header line + 4 data rows (the primary answer / deliverables[0])",
        f"The exact header line '{HEADER}' (exact column order)",
        "atomic_number and melting_point_k present as integers for all 4 elements",
        "No prose, no markdown fences, no extra/missing rows",
    ]


def get_success_criteria() -> List[str]:
    return [
        "Deliverable's first line is the exact header (no fence, no prose preamble)",
        "Exactly 4 three-column data rows",
        "Keystone: the Erbium row's melting_point_k is exactly 1802",
        "Every row carries the exact atomic number and kelvin melting point",
        "Output is ONLY the CSV (no prose / no fences / no extra whitespace)",
        "At least 2 element pages visited (the facts are page-read, not recalled)",
    ]


# ----------------------------------------------------------------------------- parse helpers ---


def _csv_lines(text: str) -> List[str]:
    """Non-empty, right-stripped lines of the whitespace-stripped deliverable (declared order).

    Drops blank lines and trailing per-line whitespace, so the keystone is tolerant of benign
    layout noise; a leading ```csv fence or a prose preamble survives as the first element (and is
    therefore caught by the header check)."""
    raw = text.strip()
    return [ln.rstrip() for ln in raw.splitlines() if ln.strip()]


def _parse_data_rows(text: str) -> Optional[List[List[str]]]:
    """If the first non-empty line is EXACTLY the header, parse the following non-empty lines as
    CSV and return them as a list of whitespace-trimmed field lists; else None.

    A ```csv fence or a prose preamble makes the first line != header -> None (keystone fails)."""
    lines = _csv_lines(text)
    if not lines or lines[0] != HEADER:
        return None
    rows: List[List[str]] = []
    for ln in lines[1:]:
        try:
            fields = next(csv.reader(io.StringIO(ln)))
        except (csv.Error, StopIteration):
            return None
        rows.append([f.strip() for f in fields])
    return rows


def _as_int(value: str) -> Optional[int]:
    """Coerce a CSV cell to an int for an equality check: a clean (optionally signed) integer after
    trimming surrounding spaces. Rejects units ('1802 K'), thousands separators ('1,802') and any
    other non-pure-integer cell (the '$'-anchor forbids trailing junk)."""
    m = re.match(r"-?\d+$", str(value).strip())
    return int(m.group()) if m else None


def _keystone_ok(result: Dict[str, Any]) -> bool:
    """KEYSTONE gate: first line is the EXACT header, EXACTLY 4 three-column data rows, AND the
    Erbium row's melting_point_k equals 1802. A fence / prose preamble / wrong header / wrong row
    count / wrong keystone value all drive it False."""
    rows = _parse_data_rows(extract_final_text(result))
    if rows is None or len(rows) != 4:
        return False
    if any(len(r) != 3 for r in rows):
        return False
    keystone_row = next(
        (r for r in rows if re.search(KEYSTONE_NAME_RX, r[0], re.I)), None
    )
    if keystone_row is None:
        return False
    return _as_int(keystone_row[2]) == KEYSTONE_VALUE


# ------------------------------------------------------------------------------- validators ---


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated effort diagnostic: the 2 facts-per-page must be page-read, not recalled."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 2, "score": min(1.0, n / 4.0),
            "reason": f"{n} visit(s) (target >=4 for four element pages; >=2 to pass)"}


def validate_keystone_csv(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): byte-exact CSV (exact header line + exactly 4 three-column rows) AND
    the Erbium row's melting_point_k == 1802.

    This single gate fuses the two things the task is really testing — that the agent produced a
    *correctly-formatted CSV at all* (no fence, no prose, right header, right row count), and that
    it read the right field off the right page. The keystone value is page-only (an obscure
    lanthanide's exact kelvin melting point), so a from-memory answer misses it.
    """
    passed = _keystone_ok(result)
    return {"check": "keystone_csv", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": ("Exact CSV header + 4 rows; Erbium melting_point_k == 1802" if passed
                       else "Deliverable is not a byte-exact CSV with Erbium melting_point_k == 1802")}


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Breadth diagnostic (UN-gated, text-based): how many of the 4 elements have BOTH verified
    fields present in the raw answer text.

    Deliberately NOT short-circuited on the keystone and NOT dependent on CSV parsing — it
    regex-scans the raw deliverable, so an agent that GATHERED the facts but mangled the format
    (prose dump, fenced block, wrong header) still scores its gathering. This is the axis that
    separates a structured agent from a linear one even when the strict-output keystone is botched.
    """
    text = extract_final_text(result).lower()
    covered = 0
    hits: List[str] = []
    for e in ENTRIES:
        has_name = bool(re.search(e["name_rx"], text))
        has_z = bool(re.search(r"\b" + str(e["atomic_number"]) + r"\b", text))
        has_mp = bool(re.search(r"\b" + str(e["melting_point_k"]) + r"\b", text))
        if has_name and has_z and has_mp:
            covered += 1
            hits.append(e["name"])
    n = len(ENTRIES)
    return {"check": "coverage", "passed": covered == n, "score": covered / n,
            "reason": f"{covered}/{n} elements with both fields gathered ({', '.join(hits) or 'none'})"}


def validate_schema(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """SECONDARY (gated on keystone): EVERY one of the 4 rows carries the EXACT atomic number AND
    the EXACT kelvin melting point (not just the keystone cell). Short-circuits to 0 when the
    keystone is absent."""
    if not _keystone_ok(result):
        return {"check": "schema", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> schema not credited"}
    rows = _parse_data_rows(extract_final_text(result)) or []
    good = 0
    for e in ENTRIES:
        row = next(
            (r for r in rows if len(r) == 3 and re.search(e["name_rx"], r[0], re.I)), None
        )
        if row is None:
            continue
        if _as_int(row[1]) == e["atomic_number"] and _as_int(row[2]) == e["melting_point_k"]:
            good += 1
    n = len(ENTRIES)
    passed = (len(rows) == 4 and good == n)
    return {"check": "schema", "passed": passed, "score": good / n,
            "reason": f"{good}/{n} rows with exact atomic number + kelvin value; row count {len(rows)}"}


def validate_strict_format(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """SECONDARY (gated on keystone): the answer is ONLY the CSV — exactly 5 lines (header + 4
    rows), the exact header, and each row byte-strict ('Element,12,3456' — no spaces around the
    commas, no trailing whitespace, no fence, no prose). A spaced-but-correct CSV keeps the
    keystone but fails HERE. Gated to 0 when the keystone is absent."""
    if not _keystone_ok(result):
        return {"check": "strict_format", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> strict-format not credited"}
    raw = extract_final_text(result).strip()
    lines = raw.splitlines()
    clean = (
        len(lines) == 1 + len(ENTRIES)
        and lines[0] == HEADER
        and all(_STRICT_ROW_RE.match(ln) for ln in lines[1:])
    )
    return {"check": "strict_format", "passed": clean, "score": 1.0 if clean else 0.0,
            "reason": ("Answer is ONLY the CSV (exact header + 4 byte-strict rows)" if clean
                       else "Answer carries prose / fences / spacing / extra rows around the CSV")}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_csv, validate_coverage,
            validate_schema, validate_strict_format]


def get_llm_validation_function() -> callable:
    # Mirror the tier-5 suite (050-057): no LLM rubric judge — every axis is a pure function of
    # the deliverable, which is exactly the point of a strict-format task.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out scaffold for the ``graph_compiled`` variant.

    Four independent leaves (one element page each) gather the two atomic cells; the AGGREGATION
    step owns the rigid CSV — it tells the cheap runtime model to emit ONLY the strict CSV with the
    EXACT header and one row per element. Encodes only STRUCTURE: it names the GIVEN elements (also
    named in the task) but leaks NO field value (no atomic number, no melting point). The id is
    keyed on the element name (a GIVEN), never on an answer value.

    Extraction hardening: the leaf instruction explicitly identifies the Wikipedia infobox format
    ('XXXX K (YYY °C, ZZZ °F)') and pins the Kelvin value as the integer BEFORE the first 'K' on
    the Melting point row — preventing the K/°C confusion that causes a model to grab the Celsius
    value (which appears in parentheses after the K value) instead of the Kelvin value.
    """
    leaves = [
        {
            "id": re.sub(r"[^a-z0-9]+", "_", e["name"].lower()).strip("_"),
            "instruction": (
                f"Open the Wikipedia page for the chemical element '{e['name']}' and read DIRECTLY "
                "from the infobox (do not guess from memory): (a) its ATOMIC NUMBER, and (b) its "
                "MELTING POINT in KELVIN. Wikipedia element infoboxes display the melting point on "
                "one line in the format 'XXXX K (YYY °C, ZZZ °F)' — the Kelvin value is "
                "the INTEGER immediately BEFORE the letter 'K' on that line (it appears first, "
                "before the parenthetical Celsius value). Report that Kelvin integer ONLY — do NOT "
                "report the °C value that appears in parentheses after the 'K'. If the page "
                "shows only °C, convert: K = round(°C + 273.15). "
                "Report: (a) the atomic number, (b) the Kelvin melting point, (c) the source URL."
            ),
            "expect": "ATOMIC NUMBER (integer) and MELTING POINT in whole Kelvin (integer) — source URL",
            "depends_on": [],
        }
        for e in ENTRIES
    ]
    return {
        "leaves": leaves,
        "aggregation": (
            "You are given, for four chemical elements, the atomic number and the melting point in "
            "whole kelvin, each with its source URL. Output ONLY a strict CSV with EXACTLY five "
            "lines: a header line that is EXACTLY 'element,atomic_number,melting_point_k', followed "
            "by EXACTLY four data rows, one per element, in that same column order. Each data row "
            "is the element name, then its atomic number (an integer), then its melting point in "
            "whole kelvin (an integer), separated by single commas with NO spaces around the "
            "commas. Do NOT include any prose, explanation, markdown code fences, units, thousands "
            "separators, a repeated header, or extra/missing rows. The first line must be the "
            "header and the last line must be the fourth data row."
        ),
    }
