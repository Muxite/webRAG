"""
Test 057: Strict structured output under research load (rule-adherence axis — NEW for the suite)
Level: integration   Weight: long   Difficulty: 7/10

The counterpoint to the chains (050/051), fan-outs (052/053) and the mixed DAG (054): those
score *what was found*; this one scores *whether the rigid output contract was honoured while
finding it*. The agent must research 3 bridges x 3 page-readable fields and emit the result as
ONLY a strict JSON array of 3 objects with EXACTLY the keys {"name","main_span_m","opened_year"}
— no prose, no markdown fences, no extra/missing keys. This is the suite's rule-adherence
discriminator, and the one maximally favouring the compiled scaffold: a cheap model that must
*reason and format at once* leaks prose, invents a key, or hallucinates a field; the compiled
plan's leaves fill atomic fields while the AGGREGATION step emits the rigid JSON deterministically
— the harness owns the format.

Bimodal-by-design:
  * KEYSTONE (0/1): the deliverable PARSES as JSON (optional ```json fence stripped first), is a
    list of dicts, and the keystone entity's keystone field is exactly correct. A prose-wrapped
    answer cannot be json.loads()'d whole -> keystone fails (format failure tanks the keystone).
  * COVERAGE (UN-gated, text-based regex over the raw answer): how many of the 3 entities x both
    fields were actually gathered — credited even when the JSON is botched, so a linear agent that
    found the facts but mangled the format still registers its gathering. This is the axis that
    separates a structured agent from a linear one when the final answer is botched.
  * SCHEMA + STRICT-FORMAT (secondaries): short-circuit to 0 when the keystone is absent —
    exactly-3-objects + exact key set + typed fields, and "ONLY JSON, no prose/markdown".

Boundary chosen (and unit-tested): the KEYSTONE tolerates a single benign ```json ... ``` fence
(strips it, then parses). The STRICT-FORMAT secondary does NOT — the task said "no markdown
fences", so a fenced-but-otherwise-clean answer keeps the keystone (1.0) yet fails strict-format
(0.0). Prose wrapping breaks both (the fence-strip is a no-op, the whole string won't parse).

Parametric-leak hardening (2026-06-26): the original trio (Akashi Kaikyo / Golden Gate / Humber)
was a set of household-name bridges whose spans a capable cheap model emits from memory with zero
visits — so the task did NOT discriminate gpt-5-mini (it scored the "research" deliverable purely
parametrically). FIX: swap to three genuinely obscure long-span suspension bridges whose exact
infobox figures are page-only, and pin the KEYSTONE on the single least-memorable, least-round
metre value. A from-memory attempt now misses the keystone span -> parametric keystone 0, while the
page-reading compiled arm reads it off the infobox -> 1.

Ground truth (verified against live English Wikipedia infoboxes, 2026-06-26):
  Hardanger Bridge      -> main span 1,310 m (4,300 ft) -> opened 2013 (17 Aug 2013)
      https://en.wikipedia.org/wiki/Hardanger_Bridge
  Osman Gazi Bridge     -> main span 1,550 m (5,090 ft) -> opened 2016 (1 Jul 2016)
      https://en.wikipedia.org/wiki/Osman_Gazi_Bridge
  Halogaland Bridge     -> main span 1,145 m (3,757 ft) -> opened 2018 (9 Dec 2018)   [KEYSTONE]
      https://en.wikipedia.org/wiki/Halogaland_Bridge   (redirect of Hålogaland Bridge)
The keystone (Halogaland main span = 1145 m) is the least-guessable value in the set: Halogaland
is an Arctic-Norway crossing (near Narvik, opened 2018) far below household-name recognition, and
1145 is the only non-round span (ends in 5, not a multiple of 10/50/100) — a model cannot recall
it, it has to be read. The infobox "Longest span" row states 1,145 m unambiguously; the only other
span figures on the page (1,345 / 1,120 m, rejected design alternatives) live in the History prose
and differ from every fixture number, so they neither flip the keystone nor false-trigger coverage.
All nine values are mutually distinct (spans 1310/1550/1145; years 2013/2016/2018; no span equals a
year), so the \\b-anchored coverage regexes never collide.
"""

from typing import Dict, Any, List, Optional
import json
import re
from agent.app.idea_test_utils import extract_final_text


# Single source of truth for the task statement, validators and the compiled plan, so they can
# never drift. ``name_rx`` matches a reported name (case-insensitive); span/year are the verified
# infobox integers; ``slug`` is the author/page slug (for documentation of provenance).
ENTRIES: List[Dict[str, Any]] = [
    {"name": "Hardanger Bridge", "name_rx": r"hardanger",
     "main_span_m": 1310, "opened_year": 2013, "slug": r"wiki/hardanger_bridge"},
    {"name": "Osman Gazi Bridge", "name_rx": r"osman\s+gazi",
     "main_span_m": 1550, "opened_year": 2016, "slug": r"wiki/osman_gazi_bridge"},
    # KEYSTONE entity. ``name_rx`` is the diacritic-free substring "logaland", which matches the
    # name whether the model renders it "Hålogaland Bridge" (real title) or ASCII "Halogaland".
    {"name": "Hålogaland Bridge", "name_rx": r"logaland",
     "main_span_m": 1145, "opened_year": 2018, "slug": r"wiki/halogaland_bridge"},
]

REQUIRED_KEYS = ("name", "main_span_m", "opened_year")

# Keystone: parse + the Halogaland Bridge's main span — the least-memorable, least-round metre
# value in the set (Arctic-Norway bridge, opened 2018) — read exactly off the infobox.
KEYSTONE_NAME_RX = r"logaland"
KEYSTONE_FIELD = "main_span_m"
KEYSTONE_VALUE = 1145


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "057",
        "test_name": "Strict JSON structured output under research load (rule adherence)",
        "difficulty_level": "7/10",
        "category": "Strict structured output (rule adherence)",
        "level": "integration",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. '{e['name']}'" for i, e in enumerate(ENTRIES, 1))
    return (
        "You are given NO URLs. For EACH of the following three bridges, search for its "
        "Wikipedia page and read two facts DIRECTLY from the page (do not guess from memory): "
        "the MAIN (longest) span in WHOLE METRES, and the YEAR it opened:\n"
        f"{listing}\n\n"
        "Return your answer as a STRICT JSON array of EXACTLY three objects, one per bridge, "
        "each object having EXACTLY these three keys (and no others):\n"
        '  "name"         -> string  (the bridge name)\n'
        '  "main_span_m"  -> integer (the main span in whole metres)\n'
        '  "opened_year"  -> integer (the year it opened)\n\n'
        "Output ONLY the JSON array. Do NOT include any prose, explanation, markdown code "
        "fences, trailing commentary, extra keys, or missing keys. The first character of your "
        "answer must be '[' and the last must be ']'.\n"
        'Example shape (illustrative values, NOT the answer): '
        '[{"name": "Some Bridge", "main_span_m": 1000, "opened_year": 1990}, ...]'
    )


def get_required_deliverables() -> List[str]:
    return [
        "A strict JSON array of exactly 3 objects (the primary answer / deliverables[0])",
        'Each object with EXACTLY the keys {"name","main_span_m","opened_year"}',
        "main_span_m and opened_year present and typed as integers for all 3 bridges",
        "No prose, no markdown fences, no extra keys",
    ]


def get_success_criteria() -> List[str]:
    return [
        "Deliverable parses as JSON (after stripping an optional ```json fence)",
        "Keystone: the Hålogaland Bridge object's main_span_m is exactly 1145",
        "Exactly 3 objects, each with the exact key set and integer-typed fields",
        "Output is ONLY the JSON array (no prose / no markdown fences)",
        "At least 2 bridge pages visited (the facts are page-read, not recalled)",
    ]


# ----------------------------------------------------------------------------- parse helpers ---

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def _strip_fences(text: str) -> str:
    """Strip a single benign ```json ... ``` (or bare ```...```) wrapper if the WHOLE deliverable
    is exactly that fence. Trailing prose after the closing fence -> no match -> returned as-is
    (so it will fail to parse, which is the intended behaviour)."""
    t = text.strip()
    m = _FENCE_RE.match(t)
    return m.group(1).strip() if m else t


def _parse_deliverable(result: Dict[str, Any]) -> Optional[Any]:
    """Fence-strip then json.loads the primary answer; return the parsed object or None."""
    raw = _strip_fences(extract_final_text(result))
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None


def _as_int(value: Any) -> Optional[int]:
    """Coerce a JSON value to an int for the (lenient) keystone equality check. Accepts a real
    int, an integral float, or a digit string like "1,145"/"1145 m"; rejects bool and anything
    non-numeric. (The STRICT-FORMAT secondary, by contrast, demands a true JSON integer.)"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        m = re.match(r"-?\d+", re.sub(r"[,\s]", "", value))
        return int(m.group()) if m else None
    return None


def _is_strict_int(value: Any) -> bool:
    """True only for a genuine JSON integer (not bool, not float, not a numeric string)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _keystone_ok(result: Dict[str, Any]) -> bool:
    """KEYSTONE gate: deliverable (fence-stripped) parses as a list of dicts AND the Halogaland
    Bridge object's main_span_m equals 1145. A prose-wrapped answer won't parse whole -> False."""
    parsed = _parse_deliverable(result)
    if not isinstance(parsed, list) or not parsed:
        return False
    if not all(isinstance(o, dict) for o in parsed):
        return False
    obj = next(
        (o for o in parsed
         if isinstance(o.get("name"), str) and re.search(KEYSTONE_NAME_RX, o["name"], re.I)),
        None,
    )
    if obj is None or KEYSTONE_FIELD not in obj:
        return False
    return _as_int(obj.get(KEYSTONE_FIELD)) == KEYSTONE_VALUE


# ------------------------------------------------------------------------------- validators ---


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated effort diagnostic: the 3 facts-per-page must be page-read, not recalled."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 2, "score": min(1.0, n / 3.0),
            "reason": f"{n} visit(s) (target >=3 for three bridge pages; >=2 to pass)"}


def validate_keystone_json(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): parses as a list of dicts AND Halogaland Bridge main_span_m == 1145.

    This single gate fuses the two things the task is really testing — that the agent produced
    *parseable structured output at all*, and that it read the right field off the right page.
    Format failure (prose) OR the wrong keystone field both drive it to 0. The keystone value is
    page-only (an obscure Arctic bridge's exact span), so a from-memory answer misses it.
    """
    passed = _keystone_ok(result)
    return {"check": "keystone_json", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": ("Parses as JSON; Halogaland Bridge main_span_m == 1145" if passed
                       else "Deliverable does not parse to JSON with Halogaland main_span_m == 1145")}


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Breadth diagnostic (UN-gated, text-based): how many of the 3 bridges have BOTH verified
    fields present in the raw answer text.

    Deliberately NOT short-circuited on the keystone and NOT dependent on the JSON parsing — it
    regex-scans the raw deliverable, so an agent that GATHERED the facts but mangled the format
    (prose dump, wrong keys) still scores its gathering. This is the axis that separates a
    structured agent from a linear one even when the strict-output keystone is botched.
    """
    text = extract_final_text(result).lower()
    covered = 0
    hits: List[str] = []
    for e in ENTRIES:
        has_name = bool(re.search(e["name_rx"], text))
        has_span = bool(re.search(r"\b" + str(e["main_span_m"]) + r"\b", text))
        has_year = bool(re.search(r"\b" + str(e["opened_year"]) + r"\b", text))
        if has_name and has_span and has_year:
            covered += 1
            hits.append(e["name"])
    n = len(ENTRIES)
    return {"check": "coverage", "passed": covered == n, "score": covered / n,
            "reason": f"{covered}/{n} bridges with both fields gathered ({', '.join(hits) or 'none'})"}


def validate_schema(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """SECONDARY (gated on keystone): exactly 3 objects, each with the EXACT key set and
    integer-typed fields. Short-circuits to 0 when the keystone is absent."""
    if not _keystone_ok(result):
        return {"check": "schema", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> schema not credited"}
    parsed = _parse_deliverable(result)
    if not isinstance(parsed, list):
        return {"check": "schema", "passed": False, "score": 0.0,
                "reason": "Deliverable is not a JSON array"}
    good = 0
    for o in parsed[:3]:
        if not isinstance(o, dict):
            continue
        if set(o.keys()) != set(REQUIRED_KEYS):
            continue
        if not isinstance(o.get("name"), str):
            continue
        if not (_is_strict_int(o.get("main_span_m")) and _is_strict_int(o.get("opened_year"))):
            continue
        good += 1
    n = len(REQUIRED_KEYS)  # 3 objects expected
    passed = (len(parsed) == 3 and good == 3)
    return {"check": "schema", "passed": passed, "score": good / n,
            "reason": f"{good}/3 objects with exact keys + integer fields; array length {len(parsed)}"}


def validate_strict_format(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """SECONDARY (gated on keystone): the answer is ONLY the JSON array — no prose, no markdown
    fences. The RAW deliverable (before any fence-strip) must start with '[' and end with ']' and
    parse as a list. A ```json-fenced answer keeps the keystone but fails HERE. Gated to 0 when
    the keystone is absent."""
    if not _keystone_ok(result):
        return {"check": "strict_format", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> strict-format not credited"}
    raw = extract_final_text(result).strip()
    clean = raw.startswith("[") and raw.endswith("]")
    if clean:
        try:
            clean = isinstance(json.loads(raw), list)
        except (ValueError, TypeError):
            clean = False
    return {"check": "strict_format", "passed": clean, "score": 1.0 if clean else 0.0,
            "reason": ("Answer is ONLY the JSON array (no prose / no fences)" if clean
                       else "Answer carries prose or markdown fences around the JSON")}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_json, validate_coverage,
            validate_schema, validate_strict_format]


def get_llm_validation_function() -> callable:
    # Mirror the tier-5 suite (050-054): no LLM rubric judge — every axis is a pure function of
    # the deliverable, which is exactly the point of a strict-format task.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out scaffold for the ``graph_compiled`` variant.

    Three independent leaves (one bridge page each) gather the two atomic fields; the AGGREGATION
    step owns the rigid JSON — it tells the cheap runtime model to emit ONLY the strict array with
    the EXACT keys. Encodes only STRUCTURE: it names the GIVEN bridges (also named in the task)
    but leaks NO field value (no span, no year) and NO key value of the keystone. The id is keyed
    on the bridge name (a given), never on an answer.
    """
    leaves = [
        {
            "id": re.sub(r"[^a-z0-9]+", "_", e["name"].lower()).strip("_"),
            "instruction": (
                f"Open the Wikipedia page for the '{e['name']}' and read DIRECTLY from the "
                "infobox (do not guess from memory): (a) its MAIN (longest) span in WHOLE "
                "METRES, and (b) the YEAR it opened."
            ),
            "expect": "MAIN SPAN in whole metres (integer) and OPENING YEAR (integer) — source URL",
            "depends_on": [],
        }
        for e in ENTRIES
    ]
    return {
        "leaves": leaves,
        "aggregation": (
            "You are given, for three bridges, the main span in metres and the opening year, each "
            "with its source URL. Output ONLY a strict JSON array of EXACTLY three objects, one "
            "per bridge, each object having EXACTLY these three keys and no others: \"name\" (the "
            "bridge name, as a string), \"main_span_m\" (the main span in whole metres, as an "
            "integer), and \"opened_year\" (the opening year, as an integer). Do NOT include any "
            "prose, explanation, markdown code fences, extra keys, or missing keys. The first "
            "character of the output must be '[' and the last must be ']'."
        ),
    }
