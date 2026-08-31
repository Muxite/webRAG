"""
Test 211: Tier 5: Derived sum over same-unit operands (combined lake max depth)
Level: graph   Weight: short   Difficulty: 6/10

WEB-GROUNDED derived-arithmetic task (evidence-ledger measurement suite). Two same-unit
operands live on TWO DIFFERENT Wikipedia pages:
  A. Lake Baikal (Siberia, Russia) -- read its MAXIMUM DEPTH, in meters.
  B. Lake Tanganyika (East Africa) -- read its MAXIMUM DEPTH, in meters.
The KEYSTONE is the SUM: Lake Baikal (Siberia, Russia) + Lake Tanganyika (East Africa) = 3,112 m. This number is PRINTED
ON NEITHER PAGE -- it exists only if the agent actually reads both operands and performs the
sum, so an ungrounded parametric guess cannot land on it by chance.

Ground truth (verified against live English Wikipedia via WebFetch, 2026-08-31):
  Lake Baikal (Siberia, Russia): 1,642 (https://en.wikipedia.org/wiki/Lake_Baikal)
  Lake Tanganyika (East Africa): 1,470 (https://en.wikipedia.org/wiki/Lake_Tanganyika)
  Lake Baikal max depth 1,642 m (deepest lake on Earth); Lake Tanganyika max depth 1,470 m (second-deepest). Neither page states the combined figure.
  SUM = 3,112 m

Margin (near-miss protection; every plausible wrong computation is decisively separated from the
+/-1% acceptance band [3,080.88, 3,143.12] around 3,112):
  * difference instead of sum: 172 -- 94% away from the correct value, decisively outside the acceptance band
  * average instead of sum: 1,556 -- 50% away from the correct value, decisively outside the acceptance band

Anti-leak: the sum value 3,112 m appears in no source page and is
never printed in the task statement or the compiled plan below -- only the two GIVEN entity
labels are named. The keystone validator additionally gates on an actual page visit
(``visit.count > 0``), so an ungrounded correct-by-luck guess still scores 0.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


OP_A: Dict[str, Any] = {
    "key": 'baikal', "label": 'Lake Baikal (Siberia, Russia)', "value": 1642.0, "url": 'https://en.wikipedia.org/wiki/Lake_Baikal',
    "fact": 'its MAXIMUM DEPTH, in meters', "slug_rx": 'wiki/lake_baikal', "name_rx": '\\bbaikal\\b',
}
OP_B: Dict[str, Any] = {
    "key": 'tanganyika', "label": 'Lake Tanganyika (East Africa)', "value": 1470.0, "url": 'https://en.wikipedia.org/wiki/Lake_Tanganyika',
    "fact": 'its MAXIMUM DEPTH, in meters', "slug_rx": 'wiki/lake_tanganyika', "name_rx": '\\btanganyika\\b',
}

# The DERIVED value -- printed on no source page; producing it is the work.
DERIVED = 3112.0
DERIVED_UNIT = 'm'
# Acceptance tolerance: +/-1% relative (absorbs rendering/rounding, never blurs into a decoy).
VALUE_TOL = 0.01

# ---------------------------------- import-time invariants ----------------------------------
# Independent second computation (a different code path than the "obvious" one) -- both must
# agree, mirroring the two-solver discipline used by the self-contained reasoning siblings
# (204/205).
_CHECK = round(DERIVED - OP_A["value"], 6)
assert abs(_CHECK - OP_B["value"]) < 1e-6, (
    "independent recomputation (subtractive-identity path DERIVED - A == B) disagrees"
)

# Margin: no decoy may fall inside (or anywhere near) the acceptance band.
_DECOYS = [('difference instead of sum', 172.0), ('average instead of sum', 1556.0)]
for _label, _decoy in _DECOYS:
    _rel = abs(_decoy - DERIVED) / abs(DERIVED)
    assert _rel > VALUE_TOL * 3, f"decoy {_label!r}={_decoy} too close to DERIVED={DERIVED}"
# The derived value must not collide with either printed raw operand (token-hygiene).
assert abs(DERIVED - OP_A["value"]) > max(abs(DERIVED), 1) * VALUE_TOL * 3
assert abs(DERIVED - OP_B["value"]) > max(abs(DERIVED), 1) * VALUE_TOL * 3


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": '211',
        "test_name": 'Tier 5: Derived sum over same-unit operands (combined lake max depth)',
        "difficulty_level": '6/10',
        "category": 'Derived arithmetic: sum/difference over same-unit operands',
        "level": "graph",
        "weight": "short",
    }


def get_task_statement() -> str:
    return (
        "You are given NO raw figures -- search to find the page(s) you need, then READ them "
        "(do not guess from memory). You need TWO values:\n"
        "  A. Open the Wikipedia page for " + OP_A["label"] + " and read " + OP_A["fact"] + ".\n"
        "  B. Open the Wikipedia page for " + OP_B["label"] + " and read " + OP_B["fact"] + ".\n\n"
        + 'Then COMPUTE the SUM of the two values you read, in m.' + "\n"
        "This computed value is NOT printed on either page -- you must do the arithmetic "
        "yourself.\n\n"
        "Report (a) the computed sum (a single number, in the correct unit -- the "
        "keystone), (b) both raw values you read with their units, citing the exact source "
        "URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The computed SUM value (the keystone)",
        "Lake Baikal (Siberia, Russia)'s raw value (with unit)",
        "Lake Tanganyika (East Africa)'s raw value (with unit)",
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
# Text helpers (house style: self-contained per module, mirrors 204/205/059)
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
    AND the agent must have actually visited at least one page (visit.count > 0) -- an
    ungrounded parametric guess earns nothing."""
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    return _has_value(_all_text(result), DERIVED)


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 1, "score": min(1.0, n / 2.0),
            "reason": f"{n} visit(s) (target 2: one page per operand; >=1 to pass)"}


def validate_keystone(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the computed SUM (3,112 m) is present, grounded."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_211", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": (f"computed SUM {DERIVED} {DERIVED_UNIT} present (grounded)" if passed
                       else f"computed SUM ({DERIVED} {DERIVED_UNIT}) missing/incorrect or ungrounded")}


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the TWO raw operands were correctly reported
    (entity name + its value). Deliberately NOT short-circuited on the keystone -- it measures
    whether the agent actually gathered both raw facts even when the final arithmetic is botched.
    Credit is CAPPED by visit count so a 0-visit parametric-memory answer banks nothing here."""
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
    GIVEN operand, each fetching ONE atomic figure. ALL arithmetic lives only in the aggregation
    step. Encodes STRUCTURE only -- it names the two GIVEN entities but leaks neither operand
    value nor the derived sum."""
    return {
        "leaves": [
            {
                "id": "a_baikal",
                "instruction": (
                    "Open the Wikipedia page for " + OP_A["label"] + " and read " + OP_A["fact"] +
                    ", directly from the page (do not guess from memory)."
                ),
                "expect": "THE VALUE (with unit) -- source URL",
                "depends_on": [],
            },
            {
                "id": "b_tanganyika",
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
            " and one for " + OP_B["label"] + ". " + 'Then COMPUTE the SUM of the two values you read, in m.' +
            " Report (a) that computed value (the keystone), (b) both raw values with their "
            "source URLs."
        ),
    }
