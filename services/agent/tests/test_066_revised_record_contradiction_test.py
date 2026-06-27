"""
Offline unit tests for the cross-source contradiction task (test 066) — free.

Cover the KEYSTONE gate (authoritative comprehensive Great Wall length 21,196.18 km /
13,170.70 mi), the UN-gated contradiction-coverage diagnostic (both sides surfaced), the
keystone-gated secondaries (flags the outdated popular value; cites the authoritative URL), and
that the compiled plan is a well-formed DAG carrying a ``verify`` leaf that templates the upstream
visit facts and leaks no correct value. Both single-line and multi-line answer layouts must score
identically. The keystone PASSES on the surveyed value, FAILS on the popular-wrong decoy
(~8,850 km / 5,500 mi), and FAILS on a vague rounded version ("about 21,000 km / 13,000 miles").
"""
from agent.app.idea_tests import test_066_revised_record_contradiction as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


# Full, correct answer — comprehensive value + flagged outdated value + authoritative citation (single line).
_FULL = (
    "The Great Wall of China's full surveyed length is 21,196.18 km (13,170.70 mi), per China's "
    "2012 comprehensive archaeological survey of all dynasties "
    "(https://en.wikipedia.org/wiki/Great_Wall_of_China). The commonly-cited figure of 8,850 km "
    "(5,500 miles) is outdated and covers only the Ming-dynasty wall, so it underestimates the total."
)

# Same content, MULTI-LINE layout — the outdated marker and the wrong value sit on DIFFERENT lines
# (no period between), exercising the newline-tolerant [^.] proximity in the identifier regex.
_FULL_MULTILINE = (
    "Comprehensive total length of the Great Wall: 21,196.18 km (13,170.70 mi), per the 2012 survey\n"
    "Source: https://en.wikipedia.org/wiki/Great_Wall_of_China\n"
    "The previously cited, now-superseded figure that covers only\n"
    "the Ming wall is 8,850 km (5,500 mi)"
)


def test_full_answer_single_line_scores_all():
    obs = {"visit": {"count": 3}}
    assert t.validate_keystone_length(_r(_FULL), obs)["score"] == 1.0
    assert t.validate_contradiction_coverage(_r(_FULL), obs)["score"] == 1.0
    assert t.validate_identifies_wrong_value(_r(_FULL), obs)["score"] == 1.0
    assert t.validate_authoritative_citation(_r(_FULL), obs)["score"] == 1.0
    assert t.validate_visits(_r(_FULL), obs)["score"] == 1.0


def test_full_answer_multiline_scores_identically():
    obs = {"visit": {"count": 3}}
    assert t.validate_keystone_length(_r(_FULL_MULTILINE), obs)["score"] == 1.0
    assert t.validate_contradiction_coverage(_r(_FULL_MULTILINE), obs)["score"] == 1.0
    assert t.validate_identifies_wrong_value(_r(_FULL_MULTILINE), obs)["score"] == 1.0
    assert t.validate_authoritative_citation(_r(_FULL_MULTILINE), obs)["score"] == 1.0


def test_wrong_keystone_asserting_popular_value_gates_secondaries():
    # The linear/parametric failure mode: confidently asserts the iconic popular value as THE answer.
    text = (
        "After checking, the claim holds: the Great Wall of China is about 8,850 km (5,500 miles) "
        "long (https://en.wikipedia.org/wiki/Great_Wall_of_China)."
    )
    obs = {"visit": {"count": 2}}
    assert not t.validate_keystone_length(_r(text), obs)["passed"]
    assert t.validate_keystone_length(_r(text), obs)["score"] == 0.0
    # UN-gated coverage still credits the one side that WAS surfaced (the popular figure).
    assert abs(t.validate_contradiction_coverage(_r(text), obs)["score"] - 0.5) < 1e-9
    # Secondaries short-circuit to 0 when the keystone is absent (bimodal).
    assert t.validate_identifies_wrong_value(_r(text), obs)["score"] == 0.0
    assert t.validate_authoritative_citation(_r(text), obs)["score"] == 0.0


def test_rounded_imprecise_value_fails_keystone():
    # A vague rounded recall ("about 21,000 km / 13,000 miles") lacks the surveyed digits and must
    # NOT pass the keystone — only the precise 21,196 / 13,170 counts.
    obs = {"visit": {"count": 2}}
    for text in (
        "The Great Wall is roughly 21,000 km long in total (about 13,000 miles).",
        "The total length is more than 21,000 kilometres.",
        "Surveys put it around 21,200 km (13,000+ mi).",
    ):
        assert not t.validate_keystone_length(_r(text), obs)["passed"], text
        assert t.validate_keystone_length(_r(text), obs)["score"] == 0.0, text


def test_correct_value_is_not_matched_as_the_wrong_value():
    # An answer carrying ONLY the comprehensive value: keystone passes, but the popular side was not
    # surfaced (no 8,850 / 5,500 token present).
    text = (
        "The Great Wall of China totals 21,196.18 km (13,170.70 mi) "
        "(https://en.wikipedia.org/wiki/Great_Wall_of_China)."
    )
    obs = {"visit": {"count": 2}}
    assert t.validate_keystone_length(_r(text), obs)["passed"]
    assert abs(t.validate_contradiction_coverage(_r(text), obs)["score"] - 0.5) < 1e-9  # only correct side
    # No outdated value flagged -> the gated identifier scores 0 even though the keystone is present.
    assert t.validate_identifies_wrong_value(_r(text), obs)["score"] == 0.0
    assert t.validate_authoritative_citation(_r(text), obs)["score"] == 1.0


def test_echoing_both_values_without_flagging_does_not_credit_identifier():
    # Lists both figures but never says which is outdated/partial -> the gated identifier must NOT fire.
    text = (
        "Great Wall of China: sources give 21,196.18 km and 8,850 km; they differ "
        "(https://en.wikipedia.org/wiki/Great_Wall_of_China)."
    )
    obs = {"visit": {"count": 2}}
    assert t.validate_keystone_length(_r(text), obs)["passed"]
    assert t.validate_contradiction_coverage(_r(text), obs)["score"] == 1.0  # both surfaced
    assert t.validate_identifies_wrong_value(_r(text), obs)["score"] == 0.0  # but not flagged


def test_partial_coverage_only_wrong_value_scores_half():
    obs = {"visit": {"count": 1}}
    text = "The commonly cited length of the Great Wall is about 8,850 km (5,500 miles)."
    assert abs(t.validate_contradiction_coverage(_r(text), obs)["score"] - 0.5) < 1e-9
    assert not t.validate_keystone_length(_r(text), obs)["passed"]


def test_no_visits_gate_zero():
    obs = {"visit": {"count": 0}}
    v = t.validate_visits(_r(_FULL), obs)
    assert v["score"] == 0.0 and not v["passed"]


def test_compiled_plan_is_wellformed_has_verify_leaf_and_leaks_nothing():
    plan = t.get_compiled_plan()
    # Well-formed DAG (no cycle / missing dep).
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 3
    assert struct["edge_count"] == 2
    assert struct["waves"] == [["comprehensive_total", "ming_only"], ["verify_popular"]]
    assert struct["edges"] == [
        "comprehensive_total->verify_popular",
        "ming_only->verify_popular",
    ]
    # The plan carries a genuine verify leaf: action=verify, details.claim=the popular claim,
    # depending on the visit leaves and templating both upstream facts.
    verify = next(l for l in plan["leaves"] if l["id"] == "verify_popular")
    assert verify["action"] == "verify"
    assert "claim" in verify["details"] and verify["details"]["claim"].strip()
    assert verify["depends_on"] == ["comprehensive_total", "ming_only"]
    assert "{comprehensive_total}" in verify["instruction"]
    assert "{ming_only}" in verify["instruction"]
    # The popular CLAIM (the value to fact-check) is present by design...
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    assert "8,850 km" in blob
    # ...but the authoritative-comprehensive value (21,196.18 / 13,170.70) leaks NOWHERE.
    for leak in ("21,196", "21196", "13,170", "13170", "21,196.18", "13,170.70"):
        assert leak not in blob, f"plan leaks the correct value token {leak!r}"


def test_task_statement_and_metadata_do_not_leak_the_answer():
    # Anti-parametric: the planted statement carries the popular claim but never the surveyed total.
    stmt = t.get_task_statement().lower()
    assert "8,850 km" in stmt  # the claim to fact-check is planted by design
    for leak in ("21,196", "21196", "13,170", "13170"):
        assert leak not in stmt, f"task statement leaks the correct value token {leak!r}"
    meta = t.get_test_metadata()
    assert meta["test_id"] == "066"
    assert meta["level"] == "integration"
