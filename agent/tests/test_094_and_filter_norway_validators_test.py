"""
Offline unit tests for the two-constraint numeric AND-filter over Norwegian fjords (test 094) — free.

Covers the AND-filter keystone gate (unique satisfier = Trondheimsfjord: length > 110 km AND max
depth < 750 m), the UN-gated coverage diagnostic (how many of the six fjords had BOTH attributes
gathered), the visit gate, the keystone-gated winner-attributes/citation secondaries, and that the
compiled plan is SIX self-describing per-fjord leaves (each leaf gathers BOTH attributes for one
fjord and restates the fjord's name) rather than twelve attribute-per-leaf leaves, and that it
leaks neither figures nor the winner.
"""
from agent.app.idea_tests import test_094_tier5_and_filter_norway as t


def _r(text):
    return {"output": {"final_deliverable": text}}


# Full enumeration table (multi-line layout): every fjord's two-way check on its own line, with
# the winner asserted at the end. Passes every validator at 1.0.
_FULL = (
    "Two-way check for all six fjords:\n"
    "- Trondheimsfjord: length = 130 km (>110 km? yes), max depth = 617 m (<750 m? yes) — "
    "https://en.wikipedia.org/wiki/Trondheimsfjord\n"
    "- Sognefjord: length = 205 km (>110 km? yes), max depth = 1,308 m (<750 m? no) — "
    "https://en.wikipedia.org/wiki/Sognefjord\n"
    "- Hardangerfjord: length = 179 km (>110 km? yes), max depth = 860 m (<750 m? no) — "
    "https://en.wikipedia.org/wiki/Hardangerfjord\n"
    "- Romsdalsfjord: length = 88 km (>110 km? no), max depth = 550 m (<750 m? yes) — "
    "https://en.wikipedia.org/wiki/Romsdalsfjord\n"
    "- Tysfjorden: length = 62 km (>110 km? no), max depth = 897 m (<750 m? no) — "
    "https://en.wikipedia.org/wiki/Tysfjorden\n"
    "- Lysefjord: length = 42 km (>110 km? no), max depth = 422 m (<750 m? yes) — "
    "https://en.wikipedia.org/wiki/Lysefjord\n"
    "The Trondheimsfjord is the only fjord satisfying both constraints."
)


def test_full_answer_scores_all():
    obs = {"visit": {"count": 6}}
    assert t.validate_keystone_filter(_r(_FULL), obs)["score"] == 1.0
    assert t.validate_coverage(_r(_FULL), obs)["score"] == 1.0
    assert t.validate_winner_attributes(_r(_FULL), obs)["score"] == 1.0
    assert t.validate_citation(_r(_FULL), obs)["score"] == 1.0
    assert t.validate_visits(_r(_FULL), obs)["score"] == 1.0


def test_single_line_answer_still_credits_keystone():
    """A compact one-line answer (no table) must still pass the keystone gate."""
    text = ("Trondheimsfjord (130 km, max depth 617 m) is the only fjord that is over 110 km long "
            "AND has a max depth under 750 m; Sognefjord, Hardangerfjord, Romsdalsfjord, Tysfjorden "
            "and Lysefjord all fail one constraint.")
    assert t.validate_keystone_filter(_r(text), {"visit": {"count": 6}})["passed"]


def test_multiline_winner_layout_still_credited():
    """Winner named on the line AFTER a header (the auto-compiled two-part layout) — the
    newline-tolerant proximity regex must still credit the keystone."""
    text = ("(a) Winner:\nThe Trondheimsfjord — the only fjord meeting both constraints.\n\n"
            "(b) All six fjords were checked against both thresholds.")
    assert t.validate_keystone_filter(_r(text), {"visit": {"count": 6}})["passed"]


def test_wrong_keystone_fails_gate_but_keeps_coverage():
    """Honest table, wrong conclusion (names Sognefjord). Keystone gate = 0, but the un-gated
    coverage diagnostic still credits all six fjords, and the keystone-gated secondaries collapse."""
    text = _FULL.replace(
        "The Trondheimsfjord is the only fjord satisfying both constraints.",
        "The Sognefjord is the only fjord satisfying both constraints.",
    )
    obs = {"visit": {"count": 6}}
    assert not t.validate_keystone_filter(_r(text), obs)["passed"]
    assert t.validate_coverage(_r(text), obs)["score"] == 1.0      # all six still gathered
    assert t.validate_citation(_r(text), obs)["score"] == 0.0      # gated on keystone
    assert t.validate_winner_attributes(_r(text), obs)["score"] == 0.0


def test_partial_coverage_scores_fraction():
    obs = {"visit": {"count": 3}}
    text = (
        "- Trondheimsfjord: length = 130 km, max depth = 617 m\n"
        "- Sognefjord: length = 205 km, max depth = 1,308 m\n"
        "- Hardangerfjord: length = 179 km, max depth = 860 m\n"
        "The Trondheimsfjord is the only one satisfying both constraints."
    )
    cov = t.validate_coverage(_r(text), obs)
    assert abs(cov["score"] - 3 / 6) < 1e-9
    assert not cov["passed"]
    assert t.validate_keystone_filter(_r(text), obs)["passed"]     # Trondheimsfjord asserted winner
    assert t.validate_visits(_r(text), obs)["score"] == 0.5


def test_no_visits_loses_visit_credit():
    obs = {"visit": {"count": 0}}
    assert t.validate_visits(_r(_FULL), obs)["score"] == 0.0        # no evidence visits


def test_ungrounded_correct_value_gates_to_zero():
    """Grounding requirement: the correct keystone VALUE/TABLE alone must NOT earn credit if the
    agent never actually visited a page (visit.count == 0) — an ungrounded parametric-memory guess
    must collapse the keystone gate (and everything gated on it) to 0, not just the value match."""
    ungrounded_obs = {"visit": {"count": 0}}
    r = _r(_FULL)
    assert t.validate_keystone_filter(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_filter(r, ungrounded_obs)["passed"] is False
    assert t.validate_winner_attributes(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citation(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_filter(r, ungrounded_obs)["score"],
        t.validate_coverage(r, ungrounded_obs)["score"],
        t.validate_winner_attributes(r, ungrounded_obs)["score"],
        t.validate_citation(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_compiled_plan_is_six_selfdescribing_leaves_and_leaks_nothing():
    plan = t.get_compiled_plan()
    leaves = plan["leaves"]
    # One leaf per fjord (six), NOT twelve attribute-per-leaf leaves.
    assert len(leaves) == len(t.ENTITIES) == 6
    assert all(leaf["depends_on"] == [] for leaf in leaves)         # pure fan-out
    assert all(leaf["id"].endswith("_attributes") for leaf in leaves)

    # Each leaf must gather BOTH attributes and RESTATE its fjord's name (the mis-bind fix):
    # the aggregation's facts_block strips leaf ids, so the fjord name must live in the answer.
    for e, leaf in zip(t.ENTITIES, leaves):
        instr = leaf["instruction"].lower()
        assert e["name"].lower() in instr, f"leaf {leaf['id']} must name its fjord"
        assert e["name"].lower() in leaf["expect"].lower(), (
            f"leaf {leaf['id']} answer format must restate the fjord name"
        )
        assert "length" in instr and "depth" in instr, (
            f"leaf {leaf['id']} must gather BOTH attributes in one leaf"
        )

    # The aggregation owns the full AND-filter (names both GIVEN thresholds).
    agg = plan["aggregation"].lower()
    assert "110" in agg and "750" in agg

    # STRUCTURE only — no attribute value and no winner assertion may leak into the plan.
    blob = (" ".join(str(leaf) for leaf in leaves) + " " + plan["aggregation"])
    for e in t.ENTITIES:
        for token in (str(e["length"]), f"{e['depth']:,}", str(e["depth"])):
            assert token not in blob, f"plan leaks {e['name']} value {token!r}"
    # Winner is never named as the answer inside the plan.
    assert not t._WINNER_WINS.search(blob), "plan must not assert Trondheimsfjord as the winner"
