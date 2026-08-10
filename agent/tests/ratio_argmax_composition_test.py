"""
Offline proof for the ``ratio_argmax`` composition kill-switch on task 088, and a pinned
counter-example for why task 079 is deliberately excluded from it.

Both tasks were flagged as sharing task 064's dual-field shape (ONE leaf reads TWO labelled
numbers off ONE page, the aggregation divides and takes the argmax). 088 genuinely does, so its
compiled plan now declares a ``ratio_argmax`` composition under the same opt-in
``IDEA_TEST_COMPILED_COMPOSITION_KILLSWITCH`` env flag that tasks 084/091 use for plain argmax.
079 does NOT, for a reason that is invisible from the plan's shape alone: its numerator's unit is
heterogeneous ACROSS items (Wikipedia prints two of the five stations' annual generation in TWh
and three in GWh, and 079's leaf instruction deliberately asks for the unit "exactly as shown"),
while ``_compose_ratio_argmax`` carries one global ``multiplier`` and one ``numerator_unit`` label
for the whole comparison. The last test here pins what that mismatch actually produces, so the
exclusion survives a future reader who only compares plan shapes.

Sibling of ``execution_compiled_ratio_argmax_validators_test.py`` (the same closed loop for 064).
"""
import pytest

from agent.app.idea_tests import test_079_tier5_ratio_argmax_b as t079
from agent.app.idea_tests import test_088_tier5_ratio_argmax_c as t088
from agent.app.testing import compiled_plan
from agent.app.testing import execution_compiled as ec

_FLAG = "IDEA_TEST_COMPILED_COMPOSITION_KILLSWITCH"
_WIKI = "https://en.wikipedia.org/wiki/"


def _slug(name: str) -> str:
    return name.replace(" ", "_")


def _fact_088(entity, discharge=None, length=None) -> str:
    """One 088 leaf's fact, shaped like real thin-leaf output: both figures explicitly labelled."""
    d = entity["discharge_m3s"] if discharge is None else discharge
    ln = entity["length_km"] if length is None else length
    return (f"Discharge: {d:,} m3/s; Length: {ln:,} km "
            f"— source: {_WIKI}{_slug(entity['name'])}")


def _results_088(unknown=()) -> dict:
    return {e["key"]: (f"UNKNOWN — {_WIKI}{_slug(e['name'])}" if e["key"] in unknown
                       else _fact_088(e))
            for e in t088.ENTITIES}


def _plan_088(monkeypatch, killswitch: bool) -> dict:
    if killswitch:
        monkeypatch.setenv(_FLAG, "1")
    else:
        monkeypatch.delenv(_FLAG, raising=False)
    return t088.get_compiled_plan()


def _score_all(module, composed: str, n_visits: int) -> dict:
    result = {"output": {"final_deliverable": composed}}
    observability = {"visit": {"count": n_visits}}
    return {v["check"]: v for v in
            (check(result, observability) for check in module.get_validation_functions())}


# ---- 088: the kill-switch is genuinely opt-in --------------------------------------------

@pytest.mark.parametrize("unset_value", ["", "0", "false", "False"])
def test_088_default_plan_declares_no_composition(monkeypatch, unset_value):
    """Off by default, and the documented off-values stay off — a benchmark task must not change
    arms because someone exported the flag as ``0``."""
    monkeypatch.setenv(_FLAG, unset_value)
    plan = t088.get_compiled_plan()
    assert "composition" not in plan and "agg_mode" not in plan


def test_088_killswitch_plan_is_the_default_plan_plus_composition(monkeypatch):
    """The A/B must isolate composition: leaves and the free-text ``aggregation`` fallback stay
    byte-identical, so the only difference between arms is the composed render."""
    off = _plan_088(monkeypatch, killswitch=False)
    on = _plan_088(monkeypatch, killswitch=True)
    assert on["leaves"] == off["leaves"]
    assert on["aggregation"] == off["aggregation"]
    assert on["agg_mode"] == "computed"
    assert set(on) - set(off) == {"agg_mode", "composition"}


def test_088_composition_is_wellformed_and_leaks_no_answer(monkeypatch):
    """Every declared item must name a real leaf, and the composition must encode STRUCTURE only —
    no discharge/length figure and no winner name beyond the five given river names."""
    plan = _plan_088(monkeypatch, killswitch=True)
    comp = plan["composition"]
    assert comp["op"] == "ratio_argmax"
    assert comp["op"] in compiled_plan.COMPOSITION_OPS   # the plan schema accepts it
    leaf_ids = {leaf["id"] for leaf in plan["leaves"]}
    assert [i["leaf"] for i in comp["items"]] == [e["key"] for e in t088.ENTITIES]
    assert all(i["leaf"] in leaf_ids for i in comp["items"])
    blob = repr(comp)
    for e in t088.ENTITIES:
        assert str(e["discharge_m3s"]) not in blob and str(e["length_km"]) not in blob
    assert f"{t088.WINNER_RATIO:.3f}"[:4] not in blob


# ---- 088: the composed render satisfies 088's own real validators -------------------------

def test_088_composed_answer_passes_every_real_validator(monkeypatch):
    plan = _plan_088(monkeypatch, killswitch=True)
    composed = ec._compose(plan["leaves"], _results_088(), plan["composition"])
    assert composed is not None
    assert composed.startswith(
        f"{t088.WINNER['name']} has the highest discharge-to-length ratio of the 5 rivers compared")
    scores = _score_all(t088, composed, len(t088.ENTITIES))
    assert all(v["score"] == 1.0 for v in scores.values()), scores


def test_088_composed_answer_beats_the_double_decoy_arithmetically(monkeypatch):
    """The winner comes out of the division, not out of the fixture text order: Mekong leads on
    BOTH raw quantities and must still lose the ratio."""
    plan = _plan_088(monkeypatch, killswitch=True)
    composed = ec._compose(plan["leaves"], _results_088(), plan["composition"])
    decoy = next(e for e in t088.ENTITIES if e["name"] == "Mekong")
    assert f"{t088.WINNER['name']}: discharge=6,500 m3/s, length=1,060 km" in composed
    assert f"discharge-to-length ratio={ec._fmt_num(round(decoy['ratio'], 3))} m3/(s*km)" in composed
    assert not composed.startswith(decoy["name"])


def test_088_composition_is_all_or_nothing_when_the_winner_is_unresolved(monkeypatch):
    """Same invariant 064 pins: dropping the deliberately-obscure winner from the comparison would
    crown a runner-up, so an unresolved item refuses the whole composition."""
    plan = _plan_088(monkeypatch, killswitch=True)
    assert ec._compose(plan["leaves"], _results_088(unknown=("fly",)), plan["composition"]) is None


def test_088_composition_refuses_a_non_winning_unresolved_item_too(monkeypatch):
    plan = _plan_088(monkeypatch, killswitch=True)
    assert ec._compose(plan["leaves"], _results_088(unknown=("salween",)), plan["composition"]) is None


# ---- 079: pinned counter-example for the exclusion ----------------------------------------

#: The composition 079 WOULD carry if it were wired like 088 — defined only here, in the test that
#: shows why it must not ship in the task's plan.
_HYPOTHETICAL_079_COMPOSITION = {
    "op": "ratio_argmax",
    "answer_noun": "station",
    "value_label": "generation-to-capacity ratio",
    "numerator_label": "generation",
    "denominator_label": "capacity",
    "numerator_unit": "GWh",
    "denominator_unit": "MW",
    "ratio_unit": "GWh/MW",
    "round_digits": 3,
    "items": [{"leaf": e["key"], "label": e["name"]} for e in t079.ENTITIES],
}


def test_079_plan_stays_free_text_only(monkeypatch):
    """079 declares no composition in EITHER arm — see its ``get_compiled_plan`` docstring."""
    for value in ("", "1"):
        monkeypatch.setenv(_FLAG, value)
        plan = t079.get_compiled_plan()
        assert "composition" not in plan and "agg_mode" not in plan


def test_079_heterogeneous_numerator_units_would_make_ratio_argmax_render_false_figures():
    """Why 079 is excluded, demonstrated rather than asserted.

    079's leaf instruction asks for the generation unit "exactly as shown" and forbids converting
    (the TWh->GWh conversion is part of what the task tests), so two of the five leaves legitimately
    return TWh. ``_compose_ratio_argmax`` has one global ``multiplier`` and one ``numerator_unit``
    label, so it divides the unconverted number and RELABELS it with the composition's unit: a
    15 TWh station renders as "generation=15 GWh ... ratio=0.005". The keystone survives here only
    by luck of these fixtures (dividing by 1,000 can only shrink a ratio, and the true winner is
    GWh-native), while ``validate_coverage`` silently loses the two TWh stations. If a future change
    adds per-item unit normalisation to the composer, this test is the one to revisit.
    """
    twh = {"wac_bennett": "15 TWh", "bratsk": "22.6 TWh"}
    results = {}
    for e in t079.ENTITIES:
        gen = twh.get(e["key"], f"{e['generation_gwh']:,} GWh")
        results[e["key"]] = (f"Generation: {gen}; Capacity: {e['capacity_mw']:,} MW "
                             f"— source: {_WIKI}{_slug(e['name'])}")

    composed = ec._compose_ratio_argmax(
        [{"id": e["key"]} for e in t079.ENTITIES], results, _HYPOTHETICAL_079_COMPOSITION)
    assert composed is not None                      # it fires -- that is precisely the problem

    # A 15 TWh station is rendered as 15 GWh, off by a factor of 1,000 in the deliverable.
    assert "W.A.C. Bennett Dam: generation=15 GWh, capacity=2,907 MW" in composed
    assert "generation-to-capacity ratio=0.005 GWh/MW" in composed

    scores = _score_all(t079, composed, len(t079.ENTITIES))
    assert scores["keystone_argmax"]["score"] == 1.0        # survives only by fixture luck
    assert scores["coverage"]["score"] == 0.6               # the two TWh stations are lost
    assert not scores["coverage"]["passed"]


def test_079_would_compose_cleanly_only_if_its_leaves_pre_normalised_the_unit():
    """The mirror image: hand the SAME composition pre-converted figures and every validator that
    the mixed-unit case broke recovers. This isolates the blocker to unit heterogeneity alone —
    079's shape really is 064's — so the fix, if ever wanted, belongs in the composer (per-item
    scaling), not in a plan tweak that would also change the free-text arm."""
    results = {
        e["key"]: (f"Generation: {e['generation_gwh']:,} GWh; Capacity: {e['capacity_mw']:,} MW "
                   f"— source: {_WIKI}{_slug(e['name'])}")
        for e in t079.ENTITIES
    }
    composed = ec._compose_ratio_argmax(
        [{"id": e["key"]} for e in t079.ENTITIES], results, _HYPOTHETICAL_079_COMPOSITION)
    scores = _score_all(t079, composed, len(t079.ENTITIES))
    assert all(v["score"] == 1.0 for v in scores.values()), scores
