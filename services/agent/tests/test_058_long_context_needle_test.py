"""
Offline unit tests for the long-context needle-in-haystack task (test 058) — free.

Cover the buried-date keystone gate (15 December 2001, day+month+year precision, single- AND
multi-line layouts), the UN-gated coverage diagnostic (the surrounding restoration facts), the
visit gate, the keystone-gated citation, a plausible-but-wrong date that fails the keystone while
coverage is retained, and that the compiled plan is a well-formed single-leaf scaffold that names
the article but leaks neither the date nor any surrounding figure.
"""
from agent.app.idea_tests import test_058_long_context_needle as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_FULL = (
    "Leaning Tower of Pisa — buried detail from the 'History following construction' section:\n"
    "After the modern stabilisation project the tower was reopened to the public on "
    "15 December 2001, and was declared stable for at least another 300 years.\n"
    "During that project the tower's tilt was reduced by 45 centimetres, returning it to its "
    "1838 position; it had been closed to the public in 1990.\n"
    "Source: https://en.wikipedia.org/wiki/Leaning_Tower_of_Pisa"
)


def test_full_answer_scores_all():
    obs = {"visit": {"count": 2}}
    assert t.validate_keystone_reopen_date(_r(_FULL), obs)["score"] == 1.0
    assert t.validate_coverage(_r(_FULL), obs)["score"] == 1.0
    assert t.validate_citation(_r(_FULL), obs)["score"] == 1.0
    assert t.validate_visits(_r(_FULL), obs)["score"] == 1.0


def test_multiline_layout_still_credits_keystone():
    """The buried date split across a header line and a value line (and comma date format) must
    still pass — the keystone match is newline-tolerant and accepts common formatting variants."""
    text = (
        "(a) Reopening date:\n"
        "December 15, 2001\n\n"
        "(b) Restoration facts (History following construction):\n"
        "- Lean reduced by 45 cm\n"
        "- Declared stable for 300 years\n"
        "- Closed to the public in 1990\n"
        "- Returned to its 1838 position\n"
        "Source: https://en.wikipedia.org/wiki/Leaning_Tower_of_Pisa"
    )
    obs = {"visit": {"count": 1}}
    assert t.validate_keystone_reopen_date(_r(text), obs)["score"] == 1.0
    assert t.validate_coverage(_r(text), obs)["score"] == 1.0
    assert t.validate_citation(_r(text), obs)["score"] == 1.0


def test_iso_date_format_passes():
    obs = {"visit": {"count": 1}}
    text = "Reopened to the public on 2001-12-15. Source: https://en.wikipedia.org/wiki/Leaning_Tower_of_Pisa"
    assert t.validate_keystone_reopen_date(_r(text), obs)["passed"]


def test_wrong_date_fails_keystone_but_keeps_coverage():
    """A plausible-but-wrong reopening date (wrong year) tanks the keystone; the surrounding
    restoration facts are still credited (un-gated); citation is gated to 0."""
    text = _FULL.replace("15 December 2001", "15 December 2011")
    obs = {"visit": {"count": 2}}
    assert not t.validate_keystone_reopen_date(_r(text), obs)["passed"]
    assert t.validate_coverage(_r(text), obs)["score"] == 1.0   # all four facts still gathered
    assert t.validate_citation(_r(text), obs)["score"] == 0.0   # citation gated on keystone


def test_bare_year_does_not_pass_keystone():
    """A parametric-style answer that only recalls the YEAR (no day/month) must NOT pass — the
    keystone demands day+month+year precision that only reading the late section yields."""
    text = ("The tower reopened after restoration in 2001. "
            "Source: https://en.wikipedia.org/wiki/Leaning_Tower_of_Pisa")
    assert not t.validate_keystone_reopen_date(_r(text), {"visit": {"count": 1}})["passed"]


def test_partial_coverage_scores_fraction():
    obs = {"visit": {"count": 1}}
    text = ("The tower was reopened to the public on 15 December 2001. The tilt was reduced by "
            "45 centimetres and it was declared stable for 300 years. "
            "Source: https://en.wikipedia.org/wiki/Leaning_Tower_of_Pisa")
    cov = t.validate_coverage(_r(text), obs)
    assert abs(cov["score"] - 2 / 4) < 1e-9   # only reduction + stability; no 1990 / 1838
    assert not cov["passed"]
    assert t.validate_keystone_reopen_date(_r(text), obs)["passed"]


def test_no_visits_loses_visit_credit():
    obs = {"visit": {"count": 0}}
    assert t.validate_keystone_reopen_date(_r(_FULL), obs)["passed"]  # parametric leak possible
    assert t.validate_visits(_r(_FULL), obs)["score"] == 0.0          # but no evidence of visiting
    assert t.validate_visits(_r(_FULL), obs)["passed"] is False


def test_keystone_absent_gates_citation_to_zero():
    """No date at all -> keystone absent -> citation secondary short-circuits to 0 even though the
    page URL is present; coverage (un-gated) is still credited."""
    text = ("The Leaning Tower of Pisa was stabilised; the lean was reduced by 45 centimetres, it "
            "was declared stable for 300 years, closed in 1990, returned to its 1838 position. "
            "Source: https://en.wikipedia.org/wiki/Leaning_Tower_of_Pisa")
    obs = {"visit": {"count": 1}}
    assert not t.validate_keystone_reopen_date(_r(text), obs)["passed"]
    assert t.validate_citation(_r(text), obs)["score"] == 0.0
    assert t.validate_coverage(_r(text), obs)["score"] == 1.0


def test_compiled_plan_is_wellformed_single_leaf_and_leaks_nothing():
    plan = t.get_compiled_plan()
    # Validates against the schema-v2 normaliser/validator without raising.
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 1
    assert struct["edge_count"] == 0
    assert struct["is_pure_fanout"] is True
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    # Names the GIVEN article and points at the buried target, but leaks NO answer value.
    blob = (" ".join(str(l) for l in plan["leaves"]) + " " + plan["aggregation"]).lower()
    assert "leaning tower of pisa" in blob
    assert "reopen" in blob and ("history following construction" in blob or "stabilis" in blob)
    for leak in ("15 december", "december 15", "2001", "45", "300", "1990", "1838", "centimetre"):
        assert leak not in blob, f"plan leaks {leak!r}"
    # The id is keyed on the GIVEN topic, never on the date answer.
    assert "2001" not in leaf["id"]
