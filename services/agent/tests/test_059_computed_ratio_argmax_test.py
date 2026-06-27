"""
Offline unit tests for the computed-ratio argmax task (test 059) — free.

Cover the keystone gate (Jan Koller = highest goals-per-appearance ratio, the computed-ratio
argmax), in single- and multi-line layout and via the deliverables[0] primary slot; the wrong
answers that shortcut to the biggest raw quantity (Miroslav Klose = most goals; Robbie Keane = most
caps) gating every credit-bearing check to zero while the UN-gated coverage diagnostic is retained;
the "lists all five without picking" case failing the keystone; the gated ratio-value check
rejecting a value outside the +/- 3% tolerance; partial coverage scoring an exact fraction; the
visit gate; and that the compiled plan is a well-formed ten-leaf pure-fan-out DAG that leaks no
figure, ratio or winner.
"""
from agent.app.idea_tests import test_059_tier5_computed_ratio_argmax as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 5}}


# Full, correct answer (single line): all ten figures, all five URLs, the winner asserted, ratio ~0.60.
_FULL_SINGLE = (
    "Goals-per-appearance ratios computed from each player's infobox national-team totals: "
    "Jan Koller scored 55 goals in 91 caps for the Czech Republic "
    "(https://en.wikipedia.org/wiki/Jan_Koller); "
    "Miroslav Klose 71 in 137 (https://en.wikipedia.org/wiki/Miroslav_Klose); "
    "Robbie Keane 68 in 146 (https://en.wikipedia.org/wiki/Robbie_Keane); "
    "Jon Dahl Tomasson 52 in 112 (https://en.wikipedia.org/wiki/Jon_Dahl_Tomasson); "
    "Henrik Larsson 37 in 106 (https://en.wikipedia.org/wiki/Henrik_Larsson). "
    "Although Klose scored the most goals and Keane won the most caps, Jan Koller has the "
    "highest goals-per-appearance ratio, about 0.60 (55/91)."
)

# Same content, MULTI-LINE layout: the winner sits under a 'Highest ... ratio:' header on its own line.
_FULL_MULTI = (
    "Goals per appearance (senior internationals), from each infobox national-team total:\n"
    "  Jan Koller — 55 goals / 91 caps — https://en.wikipedia.org/wiki/Jan_Koller\n"
    "  Miroslav Klose — 71 / 137 — https://en.wikipedia.org/wiki/Miroslav_Klose\n"
    "  Robbie Keane — 68 / 146 — https://en.wikipedia.org/wiki/Robbie_Keane\n"
    "  Jon Dahl Tomasson — 52 / 112 — https://en.wikipedia.org/wiki/Jon_Dahl_Tomasson\n"
    "  Henrik Larsson — 37 / 106 — https://en.wikipedia.org/wiki/Henrik_Larsson\n"
    "Highest goals-per-appearance ratio:\n"
    "  Jan Koller (about 0.60)\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_winner_ratio(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    # Same answer, newline-heavy layout: the keystone ("Jan Koller" under "Highest ... ratio:"),
    # all ten figures, the ratio value and all five citations must still register identically.
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_winner_ratio(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0


def test_deliverables_list_primary_slot_drives_keystone():
    # Keystone reads deliverables[0]; coverage/value/citation read the whole deliverables list.
    r = {
        "deliverables": [
            "Jan Koller has the highest goals-per-appearance ratio of the five.",
            "His ratio is about 0.60 (55 goals / 91 caps).",
            "Koller 55 in 91; Klose 71 in 137; Keane 68 in 146; Tomasson 52 in 112; "
            "Larsson 37 in 106.",
            "Sources: https://en.wikipedia.org/wiki/Jan_Koller ; "
            "https://en.wikipedia.org/wiki/Miroslav_Klose ; https://en.wikipedia.org/wiki/Robbie_Keane ; "
            "https://en.wikipedia.org/wiki/Jon_Dahl_Tomasson ; https://en.wikipedia.org/wiki/Henrik_Larsson",
        ],
        "output": {"final_deliverable": ""},
    }
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_winner_ratio(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0


def test_most_goals_shortcut_gates_to_zero():
    # The cheap-model error: shortcut to the biggest raw NUMERATOR (most goals) and name Klose.
    # Even though all five players and all ten figures appear, every credit-bearing check is 0
    # except the UN-gated coverage diagnostic.
    wrong = (
        "Miroslav Klose scored the most international goals of the five — 71 goals in 137 caps — so "
        "Klose has the best record and is the answer (https://en.wikipedia.org/wiki/Miroslav_Klose). "
        "For completeness: Jan Koller 55 goals in 91 caps "
        "(https://en.wikipedia.org/wiki/Jan_Koller); Robbie Keane 68 in 146 "
        "(https://en.wikipedia.org/wiki/Robbie_Keane); Jon Dahl Tomasson 52 in 112 "
        "(https://en.wikipedia.org/wiki/Jon_Dahl_Tomasson); Henrik Larsson 37 in 106 "
        "(https://en.wikipedia.org/wiki/Henrik_Larsson)."
    )
    r = _r(wrong)
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 0.0
    assert t.validate_winner_ratio(r, _OBS)["score"] == 0.0   # gated on keystone
    assert t.validate_citation(r, _OBS)["score"] == 0.0       # gated on keystone
    assert t.validate_coverage(r, _OBS)["score"] == 1.0       # UN-gated: ten figures retained


def test_most_caps_shortcut_gates_to_zero():
    # The other raw-quantity trap: pick the most-capped player (Robbie Keane, 146 caps), the
    # lower-ratio scorer. Winner (Koller) absent from the primary slot -> every gated check is 0.
    r = {
        "deliverables": [
            "Robbie Keane has the highest ratio (he made the most appearances, 146).",
            "Keane 146 caps, 68 goals; Klose 137/71; Koller 91/55; Tomasson 112/52; Larsson 106/37.",
            "https://en.wikipedia.org/wiki/Robbie_Keane",
        ],
        "output": {"final_deliverable": ""},
    }
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 0.0
    assert t.validate_winner_ratio(r, _OBS)["score"] == 0.0   # gated on keystone
    assert t.validate_citation(r, _OBS)["score"] == 0.0       # gated on keystone


def test_lists_all_without_picking_fails_keystone():
    # Names all five but never asserts which ratio is highest -> keystone not satisfied.
    r = _r(
        "The five forwards are Jan Koller, Miroslav Klose, Robbie Keane, Jon Dahl Tomasson and "
        "Henrik Larsson; each played senior international football."
    )
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 0.0


def test_rival_asserted_winner_in_adjacent_clause_fails_keystone():
    # Hardening: a RIVAL asserted as the winner, with Koller merely mentioned as runner-up in the
    # next clause, must NOT satisfy the keystone. The proximity window is bounded at both sentence
    # periods and clause-separating semicolons, so "the highest ratio" (Klose's) cannot reach Koller.
    for txt in (
        "Klose has the highest goals-per-game ratio; Jan Koller is second.",
        "Klose has the highest goals-per-game ratio. Jan Koller is second.",
        "Klose had a higher ratio than Jan Koller.",  # 'than Koller' -> Koller is the loser
    ):
        assert t.validate_keystone_argmax(_r(txt), _OBS)["score"] == 0.0, txt
    # ...while Koller genuinely asserted as the max still passes, even with a rival in the clause.
    for txt in (
        "The highest goals-per-appearance ratio belongs to Jan Koller.",
        "Robbie Keane has more caps than Jan Koller, but the best ratio is Koller.",
        "Jan Koller, with the highest goals-per-game ratio, is the answer.",
    ):
        assert t.validate_keystone_argmax(_r(txt), _OBS)["score"] == 1.0, txt


def test_wrong_ratio_value_outside_tolerance_zeroes_value():
    # Correct winner (keystone passes) but the ratio is computed off the wrong base (caps/goals =
    # 91/55 = 1.65), outside the +/- 3% window around 0.60. The value check zeroes; coverage stays.
    text = (
        "Jan Koller has the highest goals-per-appearance ratio of the five. I (mistakenly) "
        "computed it as 1.65 (91/55). Figures: Koller 55 goals in 91 caps; Klose 71 in 137; "
        "Keane 68 in 146; Tomasson 52 in 112; Larsson 37 in 106 "
        "(https://en.wikipedia.org/wiki/Jan_Koller)."
    )
    r = _r(text)
    assert t.validate_keystone_argmax(r, _OBS)["passed"]
    assert t.validate_winner_ratio(r, _OBS)["score"] == 0.0   # 1.65 not within 0.586-0.622
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


def test_partial_coverage_scores_fraction():
    # Only two players' figures gathered (Koller + Klose), keystone still asserted -> coverage = 2/5.
    text = (
        "Jan Koller has the highest goals-per-appearance ratio, about 0.60: he scored 55 goals "
        "in 91 caps, ahead of Klose's 71 in 137 (https://en.wikipedia.org/wiki/Jan_Koller)."
    )
    r = _r(text)
    assert abs(t.validate_coverage(r, _OBS)["score"] - 0.4) < 1e-9   # 2 of 5 players
    assert t.validate_keystone_argmax(r, _OBS)["passed"]
    assert t.validate_winner_ratio(r, _OBS)["score"] == 1.0


def test_visit_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
    assert not t.validate_visits(r, {"visit": {"count": 0}})["passed"]
    assert t.validate_visits(r, {"visit": {"count": 5}})["score"] == 1.0
    assert t.validate_visits(r, {"visit": {"count": 4}})["passed"]


def test_compiled_plan_validates_and_is_ten_leaf_fanout():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)  # must not raise (well-formed, acyclic, deps resolve)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 10
    assert struct["edge_count"] == 0
    assert struct["is_pure_fanout"] is True            # ten independent figures, one parallel wave
    assert struct["waves"] == [[
        "koller_goals", "koller_caps", "klose_goals", "klose_caps", "keane_goals", "keane_caps",
        "tomasson_goals", "tomasson_caps", "larsson_goals", "larsson_caps",
    ]]


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    # STRUCTURE only: the five GIVEN players and their national teams may appear, but no goals
    # figure, no caps figure, no ratio, and no statement of which player won.
    for leak in ("55", "91", "71", "137", "68", "146", "52", "112", "37", "106",
                 "0.60", "0.604", "0.6044", "0.518", "0.466", "0.349"):
        assert leak not in blob, f"plan leaks {leak!r}"
    # No pre-stated winner: the aggregation does not tie Koller to a 'highest ratio' assertion.
    assert not t._KOLLER_WINS.search(plan["aggregation"])
