"""
Offline unit tests for the mixed-DAG task (test 054) — free.

Cover the dependent keystone gate (master's university = Cornell), the UN-gated breadth
diagnostic (both parallel-hop authors), the keystone-gated citation, and that the compiled plan
is a genuine MIXED DAG (two independent leaves + one dependency edge, two waves) that templates
the upstream author and leaks no answer.
"""
from agent.app.idea_tests import test_054_tier5_mixed_dag as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_FULL = (
    "Author of 'Beloved': Toni Morrison (https://en.wikipedia.org/wiki/Toni_Morrison).\n"
    "Author of 'The Old Man and the Sea': Ernest Hemingway "
    "(https://en.wikipedia.org/wiki/Ernest_Hemingway).\n"
    "Toni Morrison earned her master's (MA) degree from Cornell University."
)


def test_full_answer_scores_all():
    obs = {"visit": {"count": 3}}
    assert t.validate_keystone_university(_r(_FULL), obs)["score"] == 1.0
    assert t.validate_breadth_authors(_r(_FULL), obs)["score"] == 1.0
    assert t.validate_citation(_r(_FULL), obs)["score"] == 1.0
    assert t.validate_visits(_r(_FULL), obs)["score"] == 1.0


def test_missing_keystone_keeps_breadth_but_gates_citation():
    text = _FULL.replace("from Cornell University", "from an unstated university")
    obs = {"visit": {"count": 3}}
    assert not t.validate_keystone_university(_r(text), obs)["passed"]
    assert t.validate_breadth_authors(_r(text), obs)["score"] == 1.0   # both authors still gathered
    assert t.validate_citation(_r(text), obs)["score"] == 0.0          # gated on keystone


def test_partial_breadth_scores_fraction():
    obs = {"visit": {"count": 2}}
    text = "Toni Morrison wrote Beloved and earned her MA at Cornell University."
    assert abs(t.validate_breadth_authors(_r(text), obs)["score"] - 0.5) < 1e-9  # only Morrison
    assert t.validate_keystone_university(_r(text), obs)["passed"]


def test_compiled_plan_is_a_mixed_dag_and_leaks_nothing():
    plan = t.get_compiled_plan()
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 3
    assert struct["edge_count"] == 1
    assert struct["waves"] == [["author_beloved", "author_old_man"], ["masters_university"]]
    assert struct["edges"] == ["author_beloved->masters_university"]
    # The dependent leaf templates the upstream author id.
    dep = next(l for l in plan["leaves"] if l["id"] == "masters_university")
    assert "{author_beloved}" in dep["instruction"]
    # STRUCTURE only — leaks neither author nor the university answer.
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("morrison", "hemingway", "cornell"):
        assert leak not in blob, f"plan leaks {leak}"
