"""
Offline unit tests for the multi-chain + terminal-arithmetic task (test 061) — free.

Cover the computed keystone gate (|1924 - 1971| = 47), the UN-gated breadth diagnostic (both
directors' birth years gathered, retained even when the arithmetic is botched), the
keystone-gated secondaries (each chain's director resolved + citation), correct answer in both
single- and multi-line layout, a near-miss (off-by-one year / wrong difference) that gates to
zero, a plausible parametric guess that gates to zero, and that the compiled plan is a genuine
two-chain DAG (two parallel waves, two dependency edges) that templates the upstream directors
and leaks no director / birth year / difference.
"""
from agent.app.idea_tests import test_061_tier5_director_birthyear_arithmetic as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 4}}


_FULL_SINGLE = (
    "Chain A: 'The Color of Pomegranates' -> Sergei Parajanov "
    "(https://en.wikipedia.org/wiki/Sergei_Parajanov), born 1924. "
    "Chain B: 'Silent Light' -> Carlos Reygadas "
    "(https://en.wikipedia.org/wiki/Carlos_Reygadas), born 1971. "
    "Absolute difference of the birth years = 1971 - 1924 = 47 years."
)

_FULL_MULTI = (
    "Chain A:\n"
    "  Director of 'The Color of Pomegranates': Sergei Parajanov "
    "(https://en.wikipedia.org/wiki/Sergei_Parajanov).\n"
    "  Birth year: 1924.\n"
    "Chain B:\n"
    "  Director of 'Silent Light': Carlos Reygadas "
    "(https://en.wikipedia.org/wiki/Carlos_Reygadas).\n"
    "  Birth year: 1971.\n"
    "Absolute difference between the two birth years:\n"
    "47\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_difference(r, _OBS)["score"] == 1.0
    assert t.validate_breadth_years(r, _OBS)["score"] == 1.0
    assert t.validate_chains_resolved(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    # Same answer in a newline-heavy layout: the keystone (\b47\b on its own line), the breadth
    # years, the chain directors and the citations must all still register.
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_difference(r, _OBS)["score"] == 1.0
    assert t.validate_breadth_years(r, _OBS)["score"] == 1.0
    assert t.validate_chains_resolved(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0


def test_parajanov_surname_transliteration_still_resolves():
    # The chain-resolved regex must accept the alternate transliterations of Parajanov.
    text = _FULL_SINGLE.replace("Sergei Parajanov", "Sergei Paradzhanov")
    r = _r(text)
    assert t.validate_keystone_difference(r, _OBS)["passed"]
    assert t.validate_chains_resolved(r, _OBS)["score"] == 1.0


def test_near_miss_off_by_one_year_gates_to_zero():
    # A near-miss: one birth year slipped by a year (1925 not 1924), so the computed difference is
    # 46 not 47. The keystone gate must FAIL and every keystone-gated secondary must zero out.
    near = (
        "Chain A: 'The Color of Pomegranates' -> Sergei Parajanov "
        "(https://en.wikipedia.org/wiki/Sergei_Parajanov), born 1925. "
        "Chain B: 'Silent Light' -> Carlos Reygadas "
        "(https://en.wikipedia.org/wiki/Carlos_Reygadas), born 1971. "
        "Absolute difference = 1971 - 1925 = 46 years."
    )
    r = _r(near)
    assert t.validate_keystone_difference(r, _OBS)["score"] == 0.0   # 46, not 47
    assert t.validate_chains_resolved(r, _OBS)["score"] == 0.0       # gated on keystone
    assert t.validate_citation(r, _OBS)["score"] == 0.0             # gated on keystone


def test_wrong_parametric_answer_gates_to_zero():
    # A confident-but-wrong answer: wrong birth years and a wrong difference. Even though it
    # *mentions* both directors, every credit-bearing check must be 0 (keystone absent).
    wrong = (
        "From memory: Sergei Parajanov was born in 1920 and Carlos Reygadas in 1970, "
        "so the difference between their birth years is 50 years."
    )
    r = _r(wrong)
    assert t.validate_keystone_difference(r, _OBS)["score"] == 0.0
    assert t.validate_breadth_years(r, _OBS)["score"] == 0.0      # neither real year (1924/1971)
    assert t.validate_chains_resolved(r, _OBS)["score"] == 0.0    # gated on keystone
    assert t.validate_citation(r, _OBS)["score"] == 0.0          # gated on keystone


def test_missing_keystone_keeps_breadth_but_gates_secondaries():
    # Both chains resolved (directors + years + URLs) but the terminal subtraction slipped
    # (= 48, not 47). The UN-gated breadth diagnostic stays full; the gated secondaries zero out.
    text = _FULL_SINGLE.replace("= 47 years", "= 48 years")
    r = _r(text)
    assert not t.validate_keystone_difference(r, _OBS)["passed"]
    assert t.validate_breadth_years(r, _OBS)["score"] == 1.0       # both years still gathered
    assert t.validate_chains_resolved(r, _OBS)["score"] == 0.0     # gated on keystone
    assert t.validate_citation(r, _OBS)["score"] == 0.0           # gated on keystone


def test_partial_breadth_scores_fraction():
    # Only chain A's year gathered (1924), but the difference is asserted -> keystone can pass
    # while breadth registers exactly one of two inputs.
    text = (
        "Sergei Parajanov was born in 1924 "
        "(https://en.wikipedia.org/wiki/Sergei_Parajanov); the computed difference is 47."
    )
    r = _r(text)
    assert abs(t.validate_breadth_years(r, _OBS)["score"] - 0.5) < 1e-9   # only 1924
    assert t.validate_keystone_difference(r, _OBS)["passed"]


def test_low_visits_scores_fraction():
    r = _r(_FULL_SINGLE)
    assert abs(t.validate_visits(r, {"visit": {"count": 2}})["score"] - 0.5) < 1e-9
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_ungrounded_correct_value_gates_to_zero():
    """Grounding requirement: the correct keystone VALUE STRING alone must NOT earn credit if the
    agent never actually visited a page (visit.count == 0) — an ungrounded parametric-memory guess
    of the computed difference (47) must collapse the keystone gate (and everything gated on it)
    to 0."""
    r = _r(_FULL_SINGLE)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_difference(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_difference(r, ungrounded_obs)["passed"] is False
    assert t.validate_chains_resolved(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citation(r, ungrounded_obs)["score"] == 0.0
    # UN-gated breadth is unaffected by the grounding gate (it scans raw text, not the keystone).
    assert t.validate_breadth_years(r, ungrounded_obs)["score"] == 1.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_difference(r, ungrounded_obs)["score"],
        t.validate_chains_resolved(r, ungrounded_obs)["score"],
        t.validate_citation(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_keystone_not_matched_inside_year_digits():
    # Sanity: \b47\b must not fire on the year digits or any 4-digit number; without a standalone
    # "47" the keystone is absent.
    text = (
        "Sergei Parajanov born 1924; Carlos Reygadas born 1971; "
        "films from 1969 and 2007."
    )
    r = _r(text)
    assert t.validate_keystone_difference(r, _OBS)["score"] == 0.0


def test_compiled_plan_validates_and_is_a_two_chain_dag():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)  # must not raise (well-formed, acyclic, deps resolve)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 4
    assert struct["edge_count"] == 2
    # Two parallel chains: a director wave feeding a birth-year wave.
    assert struct["waves"] == [["a_dir", "b_dir"], ["a_year", "b_year"]]
    assert struct["edges"] == ["a_dir->a_year", "b_dir->b_year"]
    # Each year leaf templates its upstream director id (real dependency edges, not duplicates).
    by_id = {l["id"]: l for l in plan["leaves"]}
    assert "{a_dir}" in by_id["a_year"]["instruction"]
    assert "{b_dir}" in by_id["b_year"]["instruction"]


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    # STRUCTURE only: names the two GIVEN films but no director, birth year, or the answer.
    for leak in ("parajanov", "paradzhanov", "paradjanov", "reygadas", "1924", "1971", "47"):
        assert leak not in blob, f"plan leaks {leak!r}"
