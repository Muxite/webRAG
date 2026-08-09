"""
Offline unit tests for the cross-source contradiction task (test 056) — free.

Cover the KEYSTONE gate (authoritative-correct current Everest height 8,848.86 m / 29,031.7 ft),
the UN-gated contradiction-coverage diagnostic (both sides surfaced), the keystone-gated secondaries
(flags the outdated popular value; cites the authoritative URL), and that the compiled plan is a
well-formed DAG carrying a ``verify`` leaf that templates the upstream visit facts and leaks no
correct value. Both single-line and multi-line answer layouts must score identically.
"""
from agent.app.idea_tests import test_056_cross_source_contradiction as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


# Full, correct answer — current value + flagged outdated value + authoritative citation (single line).
_FULL = (
    "Mount Everest's current official height is 8,848.86 m (29,031.7 ft), as jointly announced by "
    "China and Nepal in 2020 (https://en.wikipedia.org/wiki/Mount_Everest). The commonly-cited "
    "figure of 8,848 m (29,029 ft) is now outdated and was superseded by that re-survey."
)

# Same content, MULTI-LINE layout — the outdated marker and the wrong value sit on DIFFERENT lines
# (no period between), exercising the newline-tolerant [^.] proximity in the identifier regex.
_FULL_MULTILINE = (
    "Current official height of Mount Everest: 8,848.86 m (29,031.7 ft), per the 2020 survey\n"
    "Source: https://en.wikipedia.org/wiki/Mount_Everest\n"
    "The previously accepted, now-superseded value commonly cited is\n"
    "8,848 m (29,029 ft)"
)


def test_full_answer_single_line_scores_all():
    obs = {"visit": {"count": 3}}
    assert t.validate_keystone_height(_r(_FULL), obs)["score"] == 1.0
    assert t.validate_contradiction_coverage(_r(_FULL), obs)["score"] == 1.0
    assert t.validate_identifies_wrong_value(_r(_FULL), obs)["score"] == 1.0
    assert t.validate_authoritative_citation(_r(_FULL), obs)["score"] == 1.0
    assert t.validate_visits(_r(_FULL), obs)["score"] == 1.0


def test_full_answer_multiline_scores_identically():
    obs = {"visit": {"count": 3}}
    assert t.validate_keystone_height(_r(_FULL_MULTILINE), obs)["score"] == 1.0
    assert t.validate_contradiction_coverage(_r(_FULL_MULTILINE), obs)["score"] == 1.0
    assert t.validate_identifies_wrong_value(_r(_FULL_MULTILINE), obs)["score"] == 1.0
    assert t.validate_authoritative_citation(_r(_FULL_MULTILINE), obs)["score"] == 1.0


def test_wrong_keystone_asserting_popular_value_gates_secondaries():
    # The linear/parametric failure mode: confidently asserts the outdated popular value as THE answer.
    text = (
        "After checking, the claim holds: Mount Everest's official height is 8,848 m (29,029 ft) "
        "(https://en.wikipedia.org/wiki/Mount_Everest)."
    )
    obs = {"visit": {"count": 2}}
    assert not t.validate_keystone_height(_r(text), obs)["passed"]
    assert t.validate_keystone_height(_r(text), obs)["score"] == 0.0
    # UN-gated coverage still credits the one side that WAS surfaced (the popular figure).
    assert abs(t.validate_contradiction_coverage(_r(text), obs)["score"] - 0.5) < 1e-9
    # Secondaries short-circuit to 0 when the keystone is absent (bimodal).
    assert t.validate_identifies_wrong_value(_r(text), obs)["score"] == 0.0
    assert t.validate_authoritative_citation(_r(text), obs)["score"] == 0.0


def test_correct_value_is_not_matched_as_the_wrong_value():
    # An answer carrying ONLY the correct value: keystone passes, but the wrong-value side was not
    # surfaced (the 8,848 prefix inside 8,848.86 must NOT count as the popular figure).
    text = "Mount Everest is 8,848.86 m (29,031.7 ft) (https://en.wikipedia.org/wiki/Mount_Everest)."
    obs = {"visit": {"count": 2}}
    assert t.validate_keystone_height(_r(text), obs)["passed"]
    assert abs(t.validate_contradiction_coverage(_r(text), obs)["score"] - 0.5) < 1e-9  # only correct side
    # No outdated value flagged -> the gated identifier scores 0 even though the keystone is present.
    assert t.validate_identifies_wrong_value(_r(text), obs)["score"] == 0.0
    assert t.validate_authoritative_citation(_r(text), obs)["score"] == 1.0


def test_echoing_both_values_without_flagging_does_not_credit_identifier():
    # Lists both figures but never says which is outdated -> the gated identifier must NOT fire.
    text = (
        "Mount Everest: sources give 8,848.86 m and 8,848 m (29,029 ft); they differ "
        "(https://en.wikipedia.org/wiki/Mount_Everest)."
    )
    obs = {"visit": {"count": 2}}
    assert t.validate_keystone_height(_r(text), obs)["passed"]
    assert t.validate_contradiction_coverage(_r(text), obs)["score"] == 1.0  # both surfaced
    assert t.validate_identifies_wrong_value(_r(text), obs)["score"] == 0.0  # but not flagged


def test_partial_coverage_only_wrong_value_scores_half():
    obs = {"visit": {"count": 1}}
    text = "The commonly cited height of Everest is 8,848 m (29,029 ft)."
    assert abs(t.validate_contradiction_coverage(_r(text), obs)["score"] - 0.5) < 1e-9
    assert not t.validate_keystone_height(_r(text), obs)["passed"]


def test_no_visits_gate_zero():
    obs = {"visit": {"count": 0}}
    v = t.validate_visits(_r(_FULL), obs)
    assert v["score"] == 0.0 and not v["passed"]


def test_ungrounded_correct_value_gates_to_zero():
    """Grounding requirement: the correct keystone VALUE STRING alone must NOT earn credit if the
    agent never actually visited a page (visit.count == 0) — an ungrounded parametric-memory guess
    of the precise 8,848.86 figure must collapse the keystone gate (and everything gated on it) to
    0, not just the value match."""
    r = _r(_FULL)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_height(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_height(r, ungrounded_obs)["passed"] is False
    assert t.validate_identifies_wrong_value(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_authoritative_citation(r, ungrounded_obs)["score"] == 0.0
    # UN-gated coverage is unaffected by the grounding gate (it scans raw text, not the keystone).
    assert t.validate_contradiction_coverage(r, ungrounded_obs)["score"] == 1.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_height(r, ungrounded_obs)["score"],
        t.validate_identifies_wrong_value(r, ungrounded_obs)["score"],
        t.validate_authoritative_citation(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_compiled_plan_is_wellformed_has_verify_leaf_and_leaks_nothing():
    plan = t.get_compiled_plan()
    # Well-formed DAG (no cycle / missing dep).
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 3
    assert struct["edge_count"] == 2
    assert struct["waves"] == [["current_height", "previous_height"], ["verify_popular"]]
    assert struct["edges"] == [
        "current_height->verify_popular",
        "previous_height->verify_popular",
    ]
    # The plan carries a genuine verify leaf: action=verify, details.claim=the popular claim,
    # depending on the visit leaves and templating both upstream facts.
    verify = next(l for l in plan["leaves"] if l["id"] == "verify_popular")
    assert verify["action"] == "verify"
    assert "claim" in verify["details"] and verify["details"]["claim"].strip()
    assert verify["depends_on"] == ["current_height", "previous_height"]
    assert "{current_height}" in verify["instruction"]
    assert "{previous_height}" in verify["instruction"]
    # The popular CLAIM (the WRONG value to fact-check) is present by design...
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    assert "8,848 metres" in blob
    # ...but the authoritative-correct value (8,848.86 / 29,031.7) leaks NOWHERE.
    for leak in ("8,848.86", "8848.86", "29,031", "29031", "29,032", "29032"):
        assert leak not in blob, f"plan leaks the correct value token {leak!r}"
