"""
Offline unit tests for the page-only prominence argmax task (test 062) — free.

Cover the keystone gate (Jengish Chokusu = most topographically prominent peak, the page-only
prominence argmax), in single- and multi-line layout and via the deliverables[0] primary slot; the
parametric DECOY (Kongur Tagh = the elevation-tallest peak) and a different wrong peak (Mount
Gongga = prominence runner-up) gating every credit-bearing check to zero while the UN-gated coverage
diagnostic is retained; the "lists all six without picking" case failing the keystone; a rival
asserted as the winner in an adjacent clause failing the keystone; the gated prominence-value check
rejecting a value outside the +/- 2.5% band; partial coverage scoring an exact fraction; the visit
gate; and that the compiled plan is a well-formed six-leaf pure-fan-out DAG that leaks no figure or
winner.
"""
from agent.app.idea_tests import test_062_tier5_prominence_argmax as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 6}}


# Full, correct answer (single line): all six prominence figures, all six URLs, the winner asserted.
_FULL_SINGLE = (
    "Topographic prominence read from each peak's infobox: "
    "Jengish Chokusu 4,148 m (https://en.wikipedia.org/wiki/Jengish_Chokusu); "
    "Mount Gongga 3,642 m (https://en.wikipedia.org/wiki/Mount_Gongga); "
    "Kongur Tagh 3,585 m (https://en.wikipedia.org/wiki/Kongur_Tagh); "
    "Ismoil Somoni Peak 3,402 m (https://en.wikipedia.org/wiki/Ismoil_Somoni_Peak); "
    "Muztagh Ata 2,698 m (https://en.wikipedia.org/wiki/Muztagh_Ata); "
    "Noshaq 2,024 m (https://en.wikipedia.org/wiki/Noshaq). "
    "Although Kongur Tagh is the tallest by elevation, Jengish Chokusu has the highest "
    "topographic prominence, 4,148 m."
)

# Same content, MULTI-LINE layout: the winner sits under a 'Most prominent:' header on its own line.
_FULL_MULTI = (
    "Topographic prominence (metres), read from each infobox:\n"
    "  Jengish Chokusu — 4,148 m — https://en.wikipedia.org/wiki/Jengish_Chokusu\n"
    "  Mount Gongga — 3,642 m — https://en.wikipedia.org/wiki/Mount_Gongga\n"
    "  Kongur Tagh — 3,585 m — https://en.wikipedia.org/wiki/Kongur_Tagh\n"
    "  Ismoil Somoni Peak — 3,402 m — https://en.wikipedia.org/wiki/Ismoil_Somoni_Peak\n"
    "  Muztagh Ata — 2,698 m — https://en.wikipedia.org/wiki/Muztagh_Ata\n"
    "  Noshaq — 2,024 m — https://en.wikipedia.org/wiki/Noshaq\n"
    "Highest topographic prominence:\n"
    "  Jengish Chokusu (4,148 m)\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_winner_prominence(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    # Same answer, newline-heavy layout: the keystone ("Jengish Chokusu" under "Highest ... "),
    # all six figures, the prominence value and all six citations must still register identically.
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_winner_prominence(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0


def test_deliverables_list_primary_slot_drives_keystone():
    # Keystone reads deliverables[0]; coverage/value/citation read the whole deliverables list.
    r = {
        "deliverables": [
            "Jengish Chokusu has the highest topographic prominence of the six.",
            "Its prominence is 4,148 m.",
            "Jengish Chokusu 4,148 m; Mount Gongga 3,642 m; Kongur Tagh 3,585 m; "
            "Ismoil Somoni Peak 3,402 m; Muztagh Ata 2,698 m; Noshaq 2,024 m.",
            "Sources: https://en.wikipedia.org/wiki/Jengish_Chokusu ; "
            "https://en.wikipedia.org/wiki/Mount_Gongga ; https://en.wikipedia.org/wiki/Kongur_Tagh ; "
            "https://en.wikipedia.org/wiki/Ismoil_Somoni_Peak ; "
            "https://en.wikipedia.org/wiki/Muztagh_Ata ; https://en.wikipedia.org/wiki/Noshaq",
        ],
        "output": {"final_deliverable": ""},
    }
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_winner_prominence(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0


def test_elevation_decoy_gates_to_zero():
    # The cheap-model error: shortcut to the TALLEST peak (Kongur Tagh, 7,649 m elevation) and
    # name it. Even though all six peaks and all six prominence figures appear, every credit-bearing
    # check is 0 except the UN-gated coverage diagnostic.
    wrong = (
        "Kongur Tagh is the tallest of the six at 7,649 m, so it is the most prominent — Kongur "
        "Tagh has the highest topographic prominence, 3,585 m "
        "(https://en.wikipedia.org/wiki/Kongur_Tagh). "
        "For completeness: Jengish Chokusu 4,148 m "
        "(https://en.wikipedia.org/wiki/Jengish_Chokusu); Mount Gongga 3,642 m "
        "(https://en.wikipedia.org/wiki/Mount_Gongga); Ismoil Somoni Peak 3,402 m "
        "(https://en.wikipedia.org/wiki/Ismoil_Somoni_Peak); Muztagh Ata 2,698 m "
        "(https://en.wikipedia.org/wiki/Muztagh_Ata); Noshaq 2,024 m "
        "(https://en.wikipedia.org/wiki/Noshaq)."
    )
    r = _r(wrong)
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 0.0
    assert t.validate_winner_prominence(r, _OBS)["score"] == 0.0   # gated on keystone
    assert t.validate_citation(r, _OBS)["score"] == 0.0            # gated on keystone
    assert t.validate_coverage(r, _OBS)["score"] == 1.0            # UN-gated: six figures retained


def test_wrong_runnerup_peak_gates_to_zero():
    # A different wrong peak: the prominence runner-up (Mount Gongga, 3,642 m) asserted as winner.
    # Winner (Jengish) absent from the primary slot -> every gated check is 0.
    r = {
        "deliverables": [
            "Mount Gongga has the highest topographic prominence of the six.",
            "Mount Gongga 3,642 m; Kongur Tagh 3,585 m; Ismoil Somoni Peak 3,402 m; "
            "Muztagh Ata 2,698 m; Noshaq 2,024 m; Jengish Chokusu 4,148 m.",
            "https://en.wikipedia.org/wiki/Mount_Gongga",
        ],
        "output": {"final_deliverable": ""},
    }
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 0.0
    assert t.validate_winner_prominence(r, _OBS)["score"] == 0.0   # gated on keystone
    assert t.validate_citation(r, _OBS)["score"] == 0.0            # gated on keystone


def test_lists_all_without_picking_fails_keystone():
    # Names all six but never asserts which prominence is highest -> keystone not satisfied.
    r = _r(
        "The six peaks are Jengish Chokusu, Mount Gongga, Kongur Tagh, Ismoil Somoni Peak, "
        "Muztagh Ata and Noshaq; each lies in High Asia."
    )
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 0.0


def test_rival_asserted_winner_in_adjacent_clause_fails_keystone():
    # Hardening: a RIVAL asserted as the winner, with Jengish merely mentioned as runner-up in the
    # next clause, must NOT satisfy the keystone. The proximity window is bounded at both sentence
    # periods and clause-separating semicolons, so "the highest prominence" (Kongur's) cannot reach
    # Jengish.
    for txt in (
        "Kongur Tagh has the highest topographic prominence; Jengish Chokusu is second.",
        "Kongur Tagh has the highest topographic prominence. Jengish Chokusu is second.",
        "Kongur Tagh is more prominent than Jengish Chokusu.",  # 'than Jengish' -> Jengish loses
    ):
        assert t.validate_keystone_argmax(_r(txt), _OBS)["score"] == 0.0, txt
    # ...while Jengish genuinely asserted as the max still passes, even with a rival in the clause.
    for txt in (
        "The highest topographic prominence belongs to Jengish Chokusu.",
        "Kongur Tagh is taller than Jengish Chokusu, but the most prominent is Jengish Chokusu.",
        "Jengish Chokusu, with the highest topographic prominence, is the answer.",
        "Pik Pobedy has the greatest prominence of the six.",  # alias for the winner
    ):
        assert t.validate_keystone_argmax(_r(txt), _OBS)["score"] == 1.0, txt


def test_wrong_prominence_value_outside_tolerance_zeroes_value():
    # Correct winner (keystone passes) but the prominence is reported as the ELEVATION (7,439 m),
    # outside the +/- 2.5% window around 4,148. The value check zeroes; coverage stays.
    text = (
        "Jengish Chokusu has the highest topographic prominence of the six. I (mistakenly) wrote "
        "its prominence as 7,439 m. Figures: Jengish Chokusu 4,148 m; Mount Gongga 3,642 m; "
        "Kongur Tagh 3,585 m; Ismoil Somoni Peak 3,402 m; Muztagh Ata 2,698 m; Noshaq 2,024 m "
        "(https://en.wikipedia.org/wiki/Jengish_Chokusu)."
    )
    r = _r(text)
    assert t.validate_keystone_argmax(r, _OBS)["passed"]
    assert t.validate_winner_prominence(r, _OBS)["score"] == 1.0   # 4,148 still present -> in band
    # ...whereas a primary answer carrying ONLY the elevation figure (no 4,148 anywhere) zeroes it.
    only_elev = (
        "Jengish Chokusu has the highest topographic prominence; its prominence is 7,439 m "
        "(https://en.wikipedia.org/wiki/Jengish_Chokusu)."
    )
    r2 = _r(only_elev)
    assert t.validate_keystone_argmax(r2, _OBS)["passed"]
    assert t.validate_winner_prominence(r2, _OBS)["score"] == 0.0   # 7,439 not within 4,044-4,252


def test_partial_coverage_scores_fraction():
    # Only two peaks' figures gathered (Jengish + Kongur), keystone still asserted -> coverage = 2/6.
    text = (
        "Jengish Chokusu has the highest topographic prominence, 4,148 m, ahead of Kongur Tagh's "
        "3,585 m (https://en.wikipedia.org/wiki/Jengish_Chokusu)."
    )
    r = _r(text)
    assert abs(t.validate_coverage(r, _OBS)["score"] - (2.0 / 6.0)) < 1e-9   # 2 of 6 peaks
    assert t.validate_keystone_argmax(r, _OBS)["passed"]
    assert t.validate_winner_prominence(r, _OBS)["score"] == 1.0


def test_visit_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
    assert not t.validate_visits(r, {"visit": {"count": 0}})["passed"]
    assert t.validate_visits(r, {"visit": {"count": 6}})["score"] == 1.0
    assert t.validate_visits(r, {"visit": {"count": 4}})["passed"]


def test_ungrounded_correct_value_gates_to_zero():
    """Grounding requirement: the correct keystone VALUE STRING alone must NOT earn credit if the
    agent never actually visited a page (visit.count == 0) — an ungrounded parametric-memory guess
    of Jengish Chokusu as the argmax must collapse the keystone gate (and everything gated on it)
    to 0."""
    r = _r(_FULL_SINGLE)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_argmax(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_argmax(r, ungrounded_obs)["passed"] is False
    assert t.validate_winner_prominence(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citation(r, ungrounded_obs)["score"] == 0.0
    # Coverage is CAPPED BY visit count (F29 fix): a 0-visit run banks 0 coverage credit even
    # though the raw text names all six peaks, closing the "parametric recall" leak.
    assert t.validate_coverage(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_argmax(r, ungrounded_obs)["score"],
        t.validate_winner_prominence(r, ungrounded_obs)["score"],
        t.validate_citation(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_compiled_plan_validates_and_is_six_leaf_fanout():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)  # must not raise (well-formed, acyclic, deps resolve)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 6
    assert struct["edge_count"] == 0
    assert struct["is_pure_fanout"] is True            # six independent figures, one parallel wave
    assert struct["waves"] == [[
        "jengish_chokusu", "gongga", "kongur_tagh", "ismoil_somoni", "muztagh_ata", "noshaq",
    ]]


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    # STRUCTURE only: the six GIVEN peaks and their countries may appear, but no prominence figure
    # and no statement of which peak won.
    for leak in ("4,148", "4148", "3,642", "3642", "3,585", "3585", "3,402", "3402",
                 "2,698", "2698", "2,024", "2024"):
        assert leak not in blob, f"plan leaks {leak!r}"
    # No pre-stated winner: the aggregation does not tie Jengish to a 'highest prominence' assertion.
    assert not t._JENGISH_WINS.search(plan["aggregation"])
