"""
Offline unit tests for Tier 5: Derived difference over same-unit operands (chimney height gap) (test 210) -- free.

Covers the keystone gate (grounded computed absolute difference), in single- and multi-line
layout and via the deliverables[0] primary slot; a wrong/decoy value gating every
credit-bearing check to zero while the UN-gated coverage diagnostic is retained; partial
coverage scoring an exact fraction; the visit/grounding gate (an ungrounded correct-value
guess must still score 0); and that the compiled plan is a well-formed
2-leaf pure-fan-out DAG that leaks neither operand value nor the derived value (the
statement-leak test).
"""
from agent.app.idea_tests import test_210_tier5_chimney_height_difference as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 2}}


_A_VAL = t.OP_A["value"]
_B_VAL = t.OP_B["value"]


def _fmt(v):
    if float(v).is_integer():
        return f"{v:,.0f}"
    return f"{v}"


_FULL_SINGLE = (
    "GRES-2 Power Station chimney (Ekibastuz, Kazakhstan): " + _fmt(_A_VAL) + " (" + t.OP_A["url"] + "). "
    "Inco Superstack (Sudbury, Ontario, Canada): " + _fmt(_B_VAL) + " (" + t.OP_B["url"] + "). "
    "Computed absolute difference = " + _fmt(t.DERIVED) + " " + t.DERIVED_UNIT + "."
)

_FULL_MULTI = (
    "Sources:\n"
    "  GRES-2 Power Station chimney (Ekibastuz, Kazakhstan) -- " + _fmt(_A_VAL) + " -- " + t.OP_A["url"] + "\n"
    "  Inco Superstack (Sudbury, Ontario, Canada) -- " + _fmt(_B_VAL) + " -- " + t.OP_B["url"] + "\n"
    "Computed absolute difference:\n"
    "  " + _fmt(t.DERIVED) + " " + t.DERIVED_UNIT + "\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0


def test_deliverables_list_primary_slot_drives_keystone():
    r = {
        "deliverables": [
            "The computed value is " + _fmt(t.DERIVED) + " " + t.DERIVED_UNIT + ".",
            "GRES-2 Power Station chimney (Ekibastuz, Kazakhstan): " + _fmt(_A_VAL) + "; Inco Superstack (Sudbury, Ontario, Canada): " + _fmt(_B_VAL) + ".",
            "Sources: " + t.OP_A["url"] + " ; " + t.OP_B["url"],
        ],
        "output": {"final_deliverable": ""},
    }
    assert t.validate_keystone(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0


def test_wrong_decoy_value_gates_credit_bearing_checks_to_zero():
    # The engineered near-miss decoy (sum instead of difference = 800.7) reported instead of the
    # correct derived value: every credit-bearing check must gate to 0 while the UN-gated
    # coverage diagnostic (both raw operands were still gathered) is retained.
    wrong = (
        "GRES-2 Power Station chimney (Ekibastuz, Kazakhstan): " + _fmt(_A_VAL) + " (" + t.OP_A["url"] + "). "
        "Inco Superstack (Sudbury, Ontario, Canada): " + _fmt(_B_VAL) + " (" + t.OP_B["url"] + "). "
        "Computed value = " + _fmt(800.7) + " " + t.DERIVED_UNIT + " (a miscalculation)."
    )
    r = _r(wrong)
    assert t.validate_keystone(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0   # gated on keystone
    assert t.validate_coverage(r, _OBS)["score"] == 1.0    # UN-gated: both raw operands retained


def test_partial_coverage_scores_fraction():
    # Only ONE raw operand gathered (A), keystone still correctly asserted -> coverage = 1/2.
    text = (
        "GRES-2 Power Station chimney (Ekibastuz, Kazakhstan): " + _fmt(_A_VAL) + " (" + t.OP_A["url"] + "). "
        "Computed value = " + _fmt(t.DERIVED) + " " + t.DERIVED_UNIT + "."
    )
    r = _r(text)
    assert abs(t.validate_coverage(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone(r, _OBS)["passed"]


def test_visit_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
    assert not t.validate_visits(r, {"visit": {"count": 0}})["passed"]
    assert t.validate_visits(r, {"visit": {"count": 2}})["score"] == 1.0
    assert t.validate_visits(r, {"visit": {"count": 1}})["passed"]


def test_ungrounded_correct_value_gates_to_zero():
    """Grounding requirement: the correct DERIVED value alone must NOT earn credit if the agent
    never actually visited a page (visit.count == 0) -- an ungrounded parametric-memory guess
    must collapse the keystone (and everything gated on it, and coverage) to 0."""
    r = _r(_FULL_SINGLE)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone(r, ungrounded_obs)["passed"] is False
    assert t.validate_citations(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_coverage(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone(r, ungrounded_obs)["score"],
        t.validate_citations(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_compiled_plan_validates_and_is_two_leaf_fanout():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)  # must not raise (well-formed, acyclic, deps resolve)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 2
    assert struct["edge_count"] == 0
    assert struct["is_pure_fanout"] is True


def test_statement_leak_raw_values_present_derived_absent():
    """Statement-leak test: the task statement / mandate context must reference the two GIVEN
    entity labels (what the agent is told to look up) but must NEVER print either raw operand
    VALUE or the derived answer -- those must come only from actually reading the pages and
    doing the arithmetic."""
    statement = t.get_task_statement()
    assert t.OP_A["label"] in statement
    assert t.OP_B["label"] in statement
    for leaked in (_fmt(_A_VAL), _fmt(_B_VAL), _fmt(t.DERIVED)):
        assert leaked not in statement, f"task statement leaks {leaked!r}"


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    # STRUCTURE only: the two GIVEN entity labels may appear, but neither raw operand value nor
    # the derived answer may appear anywhere in the plan.
    for leaked in (_fmt(_A_VAL), _fmt(_B_VAL), _fmt(t.DERIVED)):
        assert leaked.lower() not in blob, f"plan leaks {leaked!r}"
    assert t.OP_A["label"].lower() in blob
    assert t.OP_B["label"].lower() in blob
