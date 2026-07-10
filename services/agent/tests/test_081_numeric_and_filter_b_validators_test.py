"""
Offline unit tests for the two-constraint numeric AND-filter over obscure rivers (test 081) — free.

Covers the AND-filter keystone gate (unique satisfier = Prut River: length > 800 km AND basin
< 35,000 km²), the UN-gated coverage diagnostic (how many of the six rivers had BOTH attributes
gathered), the visit gate, the keystone-gated winner-attributes/citation secondaries, and — the
point of this rework — that the compiled plan is now SIX self-describing per-river leaves (each
leaf gathers BOTH attributes for one river and restates the river's name) rather than twelve
attribute-per-leaf leaves, and that it leaks neither figures nor the winner.
"""
from agent.app.idea_tests import test_081_tier5_numeric_and_filter_b as t


def _r(text):
    return {"output": {"final_deliverable": text}}


# Full enumeration table (multi-line layout): every river's two-way check on its own line, with
# the winner asserted at the end. Passes every validator at 1.0.
_FULL = (
    "Two-way check for all six rivers:\n"
    "- Prut River: length = 953 km (>800 km? yes), basin = 27,540 km² (<35,000 km²? yes) — "
    "https://en.wikipedia.org/wiki/Prut\n"
    "- Dniester River: length = 1,362 km (>800 km? yes), basin = 68,627 km² (<35,000 km²? no) — "
    "https://en.wikipedia.org/wiki/Dniester\n"
    "- Sava River: length = 992 km (>800 km? yes), basin = 97,713 km² (<35,000 km²? no) — "
    "https://en.wikipedia.org/wiki/Sava\n"
    "- Tisza River: length = 966 km (>800 km? yes), basin = 157,186 km² (<35,000 km²? no) — "
    "https://en.wikipedia.org/wiki/Tisza\n"
    "- Vardar River: length = 388 km (>800 km? no), basin = 25,000 km² (<35,000 km²? yes) — "
    "https://en.wikipedia.org/wiki/Vardar\n"
    "- Neretva River: length = 225 km (>800 km? no), basin = 11,798 km² (<35,000 km²? yes) — "
    "https://en.wikipedia.org/wiki/Neretva\n"
    "The Prut River is the only river satisfying both constraints."
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
    text = ("Prut River (953 km, drainage basin 27,540 km²) is the only river that is over 800 km "
            "long AND has a basin under 35,000 km²; Dniester, Sava, Tisza, Vardar and Neretva all "
            "fail one constraint.")
    assert t.validate_keystone_filter(_r(text), {"visit": {"count": 6}})["passed"]


def test_multiline_winner_layout_still_credited():
    """Winner named on the line AFTER a header (the auto-compiled two-part layout) — the
    newline-tolerant proximity regex must still credit the keystone."""
    text = ("(a) Winner:\nThe Prut River — the only river meeting both constraints.\n\n"
            "(b) All six rivers were checked against both thresholds.")
    assert t.validate_keystone_filter(_r(text), {"visit": {"count": 6}})["passed"]


def test_wrong_keystone_fails_gate_but_keeps_coverage():
    """Honest table, wrong conclusion (names Dniester). Keystone gate = 0, but the un-gated
    coverage diagnostic still credits all six rivers, and the keystone-gated citation collapses."""
    text = _FULL.replace(
        "The Prut River is the only river satisfying both constraints.",
        "The Dniester River is the only river satisfying both constraints.",
    )
    obs = {"visit": {"count": 6}}
    assert not t.validate_keystone_filter(_r(text), obs)["passed"]
    assert t.validate_coverage(_r(text), obs)["score"] == 1.0      # all six still gathered
    assert t.validate_citation(_r(text), obs)["score"] == 0.0      # gated on keystone
    assert t.validate_winner_attributes(_r(text), obs)["score"] == 0.0


def test_partial_coverage_scores_fraction():
    obs = {"visit": {"count": 3}}
    text = (
        "- Prut River: length = 953 km, basin = 27,540 km²\n"
        "- Dniester River: length = 1,362 km, basin = 68,627 km²\n"
        "- Sava River: length = 992 km, basin = 97,713 km²\n"
        "The Prut River is the only one satisfying both constraints."
    )
    cov = t.validate_coverage(_r(text), obs)
    assert abs(cov["score"] - 3 / 6) < 1e-9
    assert not cov["passed"]
    assert t.validate_keystone_filter(_r(text), obs)["passed"]     # Prut asserted as winner
    assert t.validate_visits(_r(text), obs)["score"] == 0.5


def test_no_visits_loses_visit_credit():
    obs = {"visit": {"count": 0}}
    assert t.validate_keystone_filter(_r(_FULL), obs)["passed"]     # parametric leak possible
    assert t.validate_visits(_r(_FULL), obs)["score"] == 0.0        # but no evidence visits


def test_compiled_plan_is_six_selfdescribing_leaves_and_leaks_nothing():
    plan = t.get_compiled_plan()
    leaves = plan["leaves"]
    # THE REWORK: one leaf per river (six), NOT twelve attribute-per-leaf leaves.
    assert len(leaves) == len(t.ENTITIES) == 6
    assert all(leaf["depends_on"] == [] for leaf in leaves)         # pure fan-out
    assert all(leaf["id"].endswith("_attributes") for leaf in leaves)

    # Each leaf must gather BOTH attributes and RESTATE its river's name (the mis-bind fix):
    # the aggregation's facts_block strips leaf ids, so the river name must live in the answer.
    for e, leaf in zip(t.ENTITIES, leaves):
        instr = leaf["instruction"].lower()
        assert e["name"].lower() in instr, f"leaf {leaf['id']} must name its river"
        assert e["name"].lower() in leaf["expect"].lower(), (
            f"leaf {leaf['id']} answer format must restate the river name"
        )
        assert "length" in instr and "basin" in instr, (
            f"leaf {leaf['id']} must gather BOTH attributes in one leaf"
        )

    # The aggregation owns the full AND-filter.
    agg = plan["aggregation"].lower()
    assert "800" in agg and "35,000" in agg

    # STRUCTURE only — no attribute value and no winner assertion may leak into the plan.
    blob = (" ".join(str(leaf) for leaf in leaves) + " " + plan["aggregation"])
    for e in t.ENTITIES:
        for token in (str(e["length"]), f"{e['basin']:,}", str(e["basin"])):
            assert token not in blob, f"plan leaks {e['name']} value {token!r}"
    # Winner is never named as the answer inside the plan.
    assert not t._PRUT_WINS.search(blob), "plan must not assert Prut as the winner"
