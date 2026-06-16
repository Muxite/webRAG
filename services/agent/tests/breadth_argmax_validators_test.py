"""
Offline unit tests for the deepest-lake breadth/argmax task (test 053) — free.

Cover the argmax keystone gate (deepest = Baikal, 1,642 m), the UN-gated coverage diagnostic
(how many of the six lake/depth pairs were gathered), the visit gate, the keystone-gated
citations, and that the compiled plan is well-formed and leaks neither depths nor the argmax.
"""
from agent.app.idea_tests import test_053_tier5_breadth_argmax_depth as t


def _r(text):
    return {"output": {"final_deliverable": text}}


_FULL = (
    "Lake maximum depths:\n"
    "- Lake Baikal -> 1,642 m (https://en.wikipedia.org/wiki/Lake_Baikal)\n"
    "- Lake Tanganyika -> 1,470 m (https://en.wikipedia.org/wiki/Lake_Tanganyika)\n"
    "- Caspian Sea -> 1,025 m (https://en.wikipedia.org/wiki/Caspian_Sea)\n"
    "- Lake Superior -> 406 m (https://en.wikipedia.org/wiki/Lake_Superior)\n"
    "- Lake Titicaca -> 281 m (https://en.wikipedia.org/wiki/Lake_Titicaca)\n"
    "- Lake Victoria -> 81 m (https://en.wikipedia.org/wiki/Lake_Victoria)\n"
    "The deepest lake is Lake Baikal (1,642 m)."
)


def test_full_answer_scores_all():
    obs = {"visit": {"count": 6}}
    assert t.validate_keystone_deepest(_r(_FULL), obs)["score"] == 1.0
    assert t.validate_coverage(_r(_FULL), obs)["score"] == 1.0
    assert t.validate_citations(_r(_FULL), obs)["score"] == 1.0
    assert t.validate_visits(_r(_FULL), obs)["score"] == 1.0


def test_wrong_deepest_fails_keystone_but_keeps_coverage():
    text = _FULL.replace("The deepest lake is Lake Baikal (1,642 m).",
                         "The deepest lake is Lake Tanganyika.")
    obs = {"visit": {"count": 6}}
    assert not t.validate_keystone_deepest(_r(text), obs)["passed"]
    assert t.validate_coverage(_r(text), obs)["score"] == 1.0   # all six still gathered
    assert t.validate_citations(_r(text), obs)["score"] == 0.0  # citations gated on keystone


def test_partial_coverage_scores_fraction():
    obs = {"visit": {"count": 3}}
    text = ("Lake Baikal 1,642 m; Lake Tanganyika 1,470 m; Caspian Sea 1,025 m. "
            "Baikal is the deepest.")
    cov = t.validate_coverage(_r(text), obs)
    assert abs(cov["score"] - 3 / 6) < 1e-9
    assert not cov["passed"]
    assert t.validate_keystone_deepest(_r(text), obs)["passed"]  # Baikal + deepest + 1642
    assert t.validate_visits(_r(text), obs)["score"] == 0.5


def test_no_visits_loses_visit_credit():
    obs = {"visit": {"count": 0}}
    assert t.validate_keystone_deepest(_r(_FULL), obs)["passed"]  # parametric leak possible
    assert t.validate_visits(_r(_FULL), obs)["score"] == 0.0      # but no evidence visits


def test_multiline_deepest_layout_still_credited():
    """A report that puts the answer on the line AFTER the 'Deepest lake:' header must still
    score the keystone — the proximity regex tolerates the line break (regression for the
    auto-compiled plan's two-part layout)."""
    text = ("(a) Deepest Lake:\nLake Baikal: 1,642 m (Source: https://en.wikipedia.org/wiki/Lake_Baikal)\n\n"
            "(b) Maximum Depths of All Six Lakes:\nLake Baikal: 1,642 m\nLake Tanganyika: 1,470 m")
    assert t.validate_keystone_deepest(_r(text), {"visit": {"count": 6}})["passed"]


def test_listing_header_does_not_false_pass_wrong_answer():
    """Naming the WRONG deepest in (a) while merely LISTING Baikal:1,642 in (b) must NOT pass —
    'Maximum Depths' is not a superlative trigger and the (b) Baikal row is out of proximity."""
    text = ("(a) Deepest Lake:\nLake Tanganyika: 1,470 m (Source: https://en.wikipedia.org/wiki/Lake_Tanganyika)\n\n"
            "(b) Maximum Depths:\nLake Baikal: 1,642 m\nLake Tanganyika: 1,470 m")
    assert not t.validate_keystone_deepest(_r(text), {"visit": {"count": 6}})["passed"]


def test_compiled_plan_is_wellformed_and_leaks_no_answers():
    plan = t.get_compiled_plan()
    leaves = plan["leaves"]
    assert len(leaves) == len(t.ENTRIES) == 6
    assert all(leaf["depends_on"] == [] for leaf in leaves)   # pure fan-out
    assert "deepest" in plan["aggregation"].lower() or "maximum" in plan["aggregation"].lower()
    blob = " ".join(str(leaf) for leaf in leaves).lower()
    # STRUCTURE only — no depth figures and not the argmax answer.
    for e in t.ENTRIES:
        assert e["depth"].split()[0] not in blob, f"plan leaks depth {e['depth']}"
    assert "deepest" not in blob and "1,642" not in blob and "1642" not in blob
    # Every lake from the breadth set is present as a leaf target.
    for e in t.ENTRIES:
        assert e["lake"].lower() in blob, f"plan missing lake {e['lake']}"
