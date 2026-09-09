"""
Test 232: Tier 5: Derived difference over ON-PAGE DECOY operands (undersea tunnel sections)
Level: graph   Weight: short   Difficulty: 9/10

WEB-GROUNDED derived-arithmetic task (evidence-ledger measurement suite). Two same-unit operands
live on TWO DIFFERENT Wikipedia pages:
  A. Channel Tunnel (England-France) -- read the length of its UNDERSEA (under-the-sea) section, in km.
  B. Seikan Tunnel (Japan) -- read the length of its UNDERSEA (beneath-the-seabed) section, in km.
The KEYSTONE is the ABSOLUTE DIFFERENCE: |A - B| = 14.6 km. This number is PRINTED ON NEITHER
PAGE -- it exists only if the agent reads both operands and performs the difference.

WHY THIS TASK EXISTS (it is deliberately adversarial, and naturally so -- nothing here is
contrived, every decoy is text English Wikipedia actually publishes):

  1. **Same label, two numbers.** The Seikan infobox states BOTH figures under the SAME label:
     "Line length  53.85 km (33.46 mi)  23.3 km (14.5 mi) undersea". A reader keying on the field
     name alone cannot tell the total from the undersea section.
  2. **A rival entity's measurement on the subject's own page.** Seikan's body says: "the Seikan
     Tunnel is the world's longest undersea tunnel, surpassing even the Channel Tunnel (although
     the latter has a longer undersea section at 37.9 kilometres (23.5 mi) vs 23.3 kilometres
     (14.5 mi) for the Seikan Tunnel)". So operand A's value is printed on operand B's page, a
     few words from the phrase "undersea section" -- the single most attractive wrong read
     available, and it yields 0.
  3. **A total that outranks the part.** Both pages lead with their TOTAL length (53.85 km,
     50.5 km), which is the number a summary-shaped read returns.

Ground truth (verified against live English Wikipedia, 2026-09-09):
  Channel Tunnel: 37.9 (https://en.wikipedia.org/wiki/Channel_Tunnel)
    "The tunnel has the longest underwater section of any tunnel in the world, at 37.9 km
    (23.5 miles)"; restated as "37.9 km (23.5 mi) under the sea".
  Seikan Tunnel: 23.3 (https://en.wikipedia.org/wiki/Seikan_Tunnel)
    infobox "23.3 km (14.5 mi) undersea"; body "a 23.3-kilometre (14.5-mile) segment running
    beneath the seabed of the Tsugaru Strait".
  ABSOLUTE DIFFERENCE = 14.6 km

Margin (near-miss protection; every plausible wrong computation is decisively separated from the
+/-2% acceptance band [14.308, 14.892] around 14.6).

Anti-leak: 14.6 appears on no source page and is never printed in the task statement or the
compiled plan below -- only the two GIVEN entity labels are named. The keystone validator also
gates on an actual page visit (``visit.count > 0``).
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


OP_A: Dict[str, Any] = {
    "key": 'channel', "label": 'Channel Tunnel (England-France)', "value": 37.9,
    "url": 'https://en.wikipedia.org/wiki/Channel_Tunnel',
    "fact": 'the length of its UNDERSEA section -- the part running under the sea, NOT the total tunnel length',
    "slug_rx": 'wiki/channel_tunnel', "name_rx": 'channel\\s+tunnel',
}
OP_B: Dict[str, Any] = {
    "key": 'seikan', "label": 'Seikan Tunnel (Japan)', "value": 23.3,
    "url": 'https://en.wikipedia.org/wiki/Seikan_Tunnel',
    "fact": 'the length of its UNDERSEA section -- the segment beneath the seabed, NOT the total line length',
    "slug_rx": 'wiki/seikan_tunnel', "name_rx": 'seikan',
}

# The DERIVED value -- printed on no source page; producing it is the work.
DERIVED = 14.6
DERIVED_UNIT = 'km'
# Acceptance tolerance: +/-2% relative. Tighter than 212's 3% because this task's nearest decoy
# (total-minus-rival-undersea, 15.95) sits 9.2% away, and 3*2% = 6% keeps a clean gap.
VALUE_TOL = 0.02

# ---------------------------------- import-time invariants ----------------------------------
# Independent second computation (additive-identity path), as 212 does.
_CHECK = round(DERIVED + OP_B["value"], 6)
assert abs(_CHECK - OP_A["value"]) < 1e-6, (
    "independent recomputation (additive-identity path DERIVED + B == A) disagrees"
)

# Margin: no decoy may fall inside (or anywhere near) the acceptance band. Every entry is a read
# a real agent can actually make from these two pages.
_DECOYS = [
    ('both TOTAL lengths instead of undersea sections (50.5 - 53.85)', 3.35),
    ("reading Seikan's page for BOTH operands, so 37.9 - 23.3 is taken as Seikan minus itself", 0.0),
    ("Seikan TOTAL minus the Channel Tunnel undersea figure quoted on Seikan's page (53.85 - 37.9)", 15.95),
    ('sum instead of difference', 61.2),
    ("Channel Tunnel TOTAL minus Seikan undersea (50.5 - 23.3)", 27.2),
]
for _label, _decoy in _DECOYS:
    _rel = abs(_decoy - DERIVED) / abs(DERIVED)
    assert _rel > VALUE_TOL * 3, f"decoy {_label!r}={_decoy} too close to DERIVED={DERIVED}"
# The derived value must not collide with either printed raw operand (token-hygiene).
assert abs(DERIVED - OP_A["value"]) > max(abs(DERIVED), 1) * VALUE_TOL * 3
assert abs(DERIVED - OP_B["value"]) > max(abs(DERIVED), 1) * VALUE_TOL * 3


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": '232',
        "test_name": 'Tier 5: Derived difference over ON-PAGE DECOY operands (undersea tunnel sections)',
        "difficulty_level": '9/10',
        "category": 'Derived arithmetic: attribution under on-page decoys',
        "level": "graph",
        "weight": "short",
    }


def get_task_statement() -> str:
    return (
        "You are given NO raw figures -- search to find the page(s) you need, then READ them "
        "(do not guess from memory). You need TWO values:\n"
        "  A. Open the Wikipedia page for " + OP_A["label"] + " and read " + OP_A["fact"] + ".\n"
        "  B. Open the Wikipedia page for " + OP_B["label"] + " and read " + OP_B["fact"] + ".\n\n"
        + 'Then COMPUTE the ABSOLUTE DIFFERENCE between the two values you read (the larger minus the smaller), in km.' + "\n"
        "This computed value is NOT printed on either page -- you must do the arithmetic "
        "yourself.\n\n"
        "Report (a) the computed absolute difference (a single number, in the correct unit -- the "
        "keystone), (b) both raw values you read with their units, citing the exact source "
        "URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The computed ABSOLUTE DIFFERENCE value (the keystone)",
        "Channel Tunnel (England-France)'s raw undersea-section value (with unit)",
        "Seikan Tunnel (Japan)'s raw undersea-section value (with unit)",
        "Source URL(s) for every page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 1 page visited (the derived value must be computed from a real read, not recalled)",
        f"Correct computed value ({DERIVED} {DERIVED_UNIT}, within +/-{VALUE_TOL*100:.0f}%)",
        f"Both raw operand values reported ({OP_A['value']}, {OP_B['value']})",
        "Source page(s) cited",
    ]


# --------------------------------------------------------------------------------------------
# Text helpers (house style: self-contained per module, mirrors 212)
# --------------------------------------------------------------------------------------------
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


_NUM_RX = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?!\d)")


def _numbers(text: str) -> List[float]:
    out: List[float] = []
    for whole, frac in _NUM_RX.findall(text):
        try:
            out.append(float(whole.replace(",", "") + ("." + frac if frac else "")))
        except ValueError:
            continue
    return out


def _has_value(text: str, target: float, tol: float = VALUE_TOL) -> bool:
    nums = _numbers(text)
    return any(abs(v - target) <= abs(target) * tol for v in nums)


def _has_operand(text: str, op: Dict[str, Any]) -> bool:
    return bool(re.search(op["name_rx"], text, re.IGNORECASE)) and _has_value(text, op["value"], tol=0.005)


# --------------------------------------------------------------------------------------------
# Validators
# --------------------------------------------------------------------------------------------
def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE gate (hard 0/1): the DERIVED value, printed on no source page, must be present
    AND the agent must have actually visited at least one page (visit.count > 0)."""
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    return _has_value(_all_text(result), DERIVED)


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 1, "score": min(1.0, n / 2.0),
            "reason": f"{n} visit(s) (target 2: one page per operand; >=1 to pass)"}


def validate_keystone(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the computed ABSOLUTE DIFFERENCE (14.6 km) is present, grounded."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_232", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": (f"computed ABSOLUTE DIFFERENCE {DERIVED} {DERIVED_UNIT} present (grounded)" if passed
                       else f"computed ABSOLUTE DIFFERENCE ({DERIVED} {DERIVED_UNIT}) missing/incorrect or ungrounded")}


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the TWO raw operands were correctly reported.
    On this task it is also the decoy detector -- an agent that reports 37.9 for BOTH tunnels
    (the trap sentence on Seikan's page) scores 1/2 here, not 2/2."""
    text = _all_text(result)
    hits = [op["label"] for op in (OP_A, OP_B) if _has_operand(text, op)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(hits), max(n_visits, 0)) if n_visits > 0 else 0
    n = 2
    return {"check": "coverage", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} raw operand(s) gathered ({', '.join(hits[:credited]) or 'none'})"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the source page(s). Short-circuits to 0 without the keystone."""
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = bool(re.search(OP_A["slug_rx"], text)) or bool(re.search(OP_B["slug_rx"], text))
    return {"check": "citations", "passed": cited, "score": 1.0 if cited else 0.0,
            "reason": f"source page cited={cited}"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone,
        validate_coverage,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    # Deterministic-only: no LLM judge (validator_lint [LLM] severity).
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored 2-leaf fan-out scaffold for the ``graph_compiled`` variant: one leaf per
    GIVEN operand. ALL arithmetic lives only in the aggregation step. Encodes STRUCTURE only --
    it names the two GIVEN entities but leaks neither operand value nor the derived difference."""
    return {
        "leaves": [
            {
                "id": "a_channel",
                "instruction": (
                    "Open the Wikipedia page for " + OP_A["label"] + " and read " + OP_A["fact"] +
                    ", directly from the page (do not guess from memory)."
                ),
                "expect": "THE VALUE (with unit) -- source URL",
                "depends_on": [],
            },
            {
                "id": "b_seikan",
                "instruction": (
                    "Open the Wikipedia page for " + OP_B["label"] + " and read " + OP_B["fact"] +
                    ", directly from the page (do not guess from memory)."
                ),
                "expect": "THE VALUE (with unit) -- source URL",
                "depends_on": [],
            },
        ],
        "aggregation": (
            "You now have two values, each read from a source page: one for " + OP_A["label"] +
            " and one for " + OP_B["label"] + ". " + 'Then COMPUTE the ABSOLUTE DIFFERENCE between the two values you read (the larger minus the smaller), in km.' +
            " Report (a) that computed value (the keystone), (b) both raw values with their "
            "source URLs."
        ),
    }
