"""
Offline unit tests for the page-only bridge-span argmax task (test 077) — free.

Covers the keystone gate (Russky Bridge = longest main span of six bridges) in single- and
multi-line layout; the fame-decoy trap (Pont de Normandie, only 4th by span) gating every
credit-bearing check to zero while the UN-gated coverage diagnostic is retained; the grounding
requirement (a correct-value answer with zero visits must not earn credit); partial coverage
scoring an exact fraction; the visit process metric; and the compiled plan being a well-formed
6-way independent fan-out that leaks no span figure or winner.
"""
from agent.app.idea_tests import test_077_tier5_pageonly_argmax as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 6}}

_FULL_SINGLE = (
    "Russky Bridge 1,104 m; Edong Yangtze River Bridge 926 m; Tatara Bridge 890 m; "
    "Pont de Normandie 856 m; Rion-Antirion Bridge 560 m; Helgeland Bridge 425 m. "
    "Russky Bridge has the longest main span of the six at 1,104 m. Sources: "
    "https://en.wikipedia.org/wiki/Russky_Bridge https://en.wikipedia.org/wiki/Edong_Yangtze_River_Bridge "
    "https://en.wikipedia.org/wiki/Tatara_Bridge https://en.wikipedia.org/wiki/Pont_de_Normandie "
    "https://en.wikipedia.org/wiki/Rion-Antirion_Bridge https://en.wikipedia.org/wiki/Helgeland_Bridge"
)

_FULL_MULTI = (
    "Spans (m):\n"
    "  Russky Bridge - 1104 - https://en.wikipedia.org/wiki/Russky_Bridge\n"
    "  Edong Yangtze River Bridge - 926 - https://en.wikipedia.org/wiki/Edong_Yangtze_River_Bridge\n"
    "  Tatara Bridge - 890 - https://en.wikipedia.org/wiki/Tatara_Bridge\n"
    "  Pont de Normandie - 856 - https://en.wikipedia.org/wiki/Pont_de_Normandie\n"
    "  Rion-Antirion Bridge - 560 - https://en.wikipedia.org/wiki/Rion-Antirion_Bridge\n"
    "  Helgeland Bridge - 425 - https://en.wikipedia.org/wiki/Helgeland_Bridge\n"
    "Longest span:\n"
    "  Russky Bridge\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_winner_span(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_winner_span(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0


def test_fame_decoy_gates_to_zero_but_keeps_coverage():
    wrong = _FULL_MULTI.replace(
        "Longest span:\n  Russky Bridge\n",
        "Most famous (and longest, in my view):\n  Pont de Normandie\n",
    )
    r = _r(wrong)
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 0.0
    assert t.validate_winner_span(r, _OBS)["score"] == 0.0
    assert t.validate_citation(r, _OBS)["score"] == 0.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


def test_ungrounded_correct_answer_gates_to_zero():
    """Grounding requirement: the correct keystone WINNER alone must NOT earn credit if the agent
    never actually visited a page (visit.count == 0) — an ungrounded parametric-memory guess must
    collapse the keystone gate (and everything gated on it) to 0."""
    r = _r(_FULL_SINGLE)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_argmax(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_argmax(r, ungrounded_obs)["passed"] is False
    assert t.validate_winner_span(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citation(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_argmax(r, ungrounded_obs)["score"],
        t.validate_coverage(r, ungrounded_obs)["score"],
        t.validate_winner_span(r, ungrounded_obs)["score"],
        t.validate_citation(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_partial_coverage_scores_fraction():
    partial = "Russky Bridge 1104 m. Tatara Bridge 890 m."
    r = _r(partial)
    cov = t.validate_coverage(r, _OBS)
    assert abs(cov["score"] - 2 / 6) < 1e-9


def test_visit_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_visits(r, {"visit": {"count": 0}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 6}})["score"] == 1.0


def test_compiled_plan_is_well_formed_six_way_fanout():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 6
    assert struct["edge_count"] == 0


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("1104", "926", "890", "856", "560", "425"):
        assert leak not in blob, f"plan leaks {leak!r}"
