"""
Offline unit tests for the compiled-plan DAG schema (testing/compiled_plan.py) — free.

Cover normalization (defaults, id slugging, deps coercion), strict validation (missing dep,
self-dep, cycle, empty), the topological wave grouping for the three task shapes (pure fan-out,
dependent chain, mixed DAG), one-level ``{dep_id}`` substitution, and the structural summary.
"""
import pytest

from agent.app.testing import compiled_plan as cp
from agent.app.testing.compiled_plan import PlanValidationError


def test_normalize_defaults_and_id_slug():
    plan = cp.normalize_plan({"leaves": [{"instruction": "Find the Author of X"}]})
    leaf = plan["leaves"][0]
    assert leaf["id"] == "find_the_author_of_x"   # slugged from instruction
    assert leaf["depends_on"] == []
    assert leaf["expect"] == ""
    assert plan["aggregation"]                     # default aggregation filled in


def test_normalize_coerces_string_depends_on():
    plan = cp.normalize_plan({"leaves": [
        {"id": "a", "instruction": "a"},
        {"id": "b", "instruction": "b", "depends_on": "a"},  # string, not list
    ]})
    assert plan["leaves"][1]["depends_on"] == ["a"]


def test_validate_rejects_empty():
    with pytest.raises(PlanValidationError):
        cp.validate_plan({"leaves": []})


def test_validate_rejects_missing_dep():
    with pytest.raises(PlanValidationError):
        cp.validate_plan({"leaves": [{"id": "a", "instruction": "a", "depends_on": ["ghost"]}]})


def test_validate_rejects_self_dep():
    with pytest.raises(PlanValidationError):
        cp.validate_plan({"leaves": [{"id": "a", "instruction": "a", "depends_on": ["a"]}]})


def test_validate_rejects_duplicate_ids():
    with pytest.raises(PlanValidationError):
        cp.validate_plan({"leaves": [
            {"id": "a", "instruction": "x"}, {"id": "a", "instruction": "y"}]})


def test_validate_rejects_cycle():
    with pytest.raises(PlanValidationError):
        cp.validate_plan({"leaves": [
            {"id": "a", "instruction": "a", "depends_on": ["b"]},
            {"id": "b", "instruction": "b", "depends_on": ["a"]},
        ]})


def test_waves_pure_fanout_is_single_wave():
    leaves = cp.normalize_plan({"leaves": [
        {"id": "a", "instruction": "a"}, {"id": "b", "instruction": "b"},
        {"id": "c", "instruction": "c"}]})["leaves"]
    assert cp.topological_waves(leaves) == [["a", "b", "c"]]


def test_waves_dependent_chain_is_one_per_wave():
    leaves = cp.normalize_plan({"leaves": [
        {"id": "a", "instruction": "a"},
        {"id": "b", "instruction": "b", "depends_on": ["a"]},
        {"id": "c", "instruction": "c", "depends_on": ["b"]},
    ]})["leaves"]
    assert cp.topological_waves(leaves) == [["a"], ["b"], ["c"]]


def test_waves_mixed_dag():
    # Two independent leaves, then one dependent on the first -> [[a,b],[c]].
    leaves = cp.normalize_plan({"leaves": [
        {"id": "a", "instruction": "a"},
        {"id": "b", "instruction": "b"},
        {"id": "c", "instruction": "c", "depends_on": ["a"]},
    ]})["leaves"]
    assert cp.topological_waves(leaves) == [["a", "b"], ["c"]]


def test_substitute_only_known_ids_and_keeps_other_braces():
    out = cp.substitute_deps("The author is {a}. Format: {not_a_dep}", {"a": "Toni Morrison"})
    assert out == "The author is Toni Morrison. Format: {not_a_dep}"


def test_substitute_noop_without_deps():
    assert cp.substitute_deps("plain {x}", {}) == "plain {x}"


def test_plan_structure_reports_shape():
    s = cp.plan_structure({"leaves": [
        {"id": "a", "instruction": "a"},
        {"id": "b", "instruction": "b"},
        {"id": "c", "instruction": "c", "depends_on": ["a"]},
    ], "aggregation": "merge"})
    assert s["leaf_count"] == 3
    assert s["edge_count"] == 1
    assert s["edges"] == ["a->c"]
    assert s["waves"] == [["a", "b"], ["c"]]
    assert s["wave_widths"] == [2, 1]
    assert s["is_pure_fanout"] is False
    assert s["is_dag_chain"] is False


def test_plan_structure_pure_fanout_flag():
    s = cp.plan_structure({"leaves": [{"id": "a", "instruction": "a"}, {"id": "b", "instruction": "b"}]})
    assert s["is_pure_fanout"] is True
    assert s["edge_count"] == 0


# --- agg_mode / composition promotion (byte-identical for plans that don't set them) -----------
def test_normalize_omits_agg_mode_and_composition_when_absent():
    plan = cp.normalize_plan({"leaves": [{"id": "a", "instruction": "a"}]})
    assert plan["agg_mode"] is None
    assert plan["composition"] is None


def test_normalize_agg_mode_lowercased_and_stripped():
    plan = cp.normalize_plan({"leaves": [{"id": "a", "instruction": "a"}], "agg_mode": "  COMPUTED  "})
    assert plan["agg_mode"] == "computed"


def test_normalize_blank_agg_mode_is_none():
    plan = cp.normalize_plan({"leaves": [{"id": "a", "instruction": "a"}], "agg_mode": "   "})
    assert plan["agg_mode"] is None


def test_normalize_composition_round_trips_when_dict_shaped():
    composition = {"op": "argmax", "answer_noun": "peak", "value_label": "elevation",
                    "items": [{"leaf": "a", "label": "Peak A", "type": "number"}]}
    plan = cp.normalize_plan({
        "leaves": [{"id": "a", "instruction": "a"}],
        "agg_mode": "computed",
        "composition": composition,
    })
    assert plan["composition"] == composition


def test_normalize_composition_ignored_when_not_dict_shaped():
    plan = cp.normalize_plan({
        "leaves": [{"id": "a", "instruction": "a"}],
        "agg_mode": "computed",
        "composition": "not-a-dict",
    })
    assert plan["composition"] is None


def test_composition_ops_matches_execution_compiled_composers():
    # execution_compiled.py can't import compiled_plan back (circular), so COMPOSITION_OPS is a
    # hand-kept mirror of execution_compiled._COMPOSERS' keys — this test is the tripwire that
    # catches drift if a composer is ever added/removed on one side without the other.
    from agent.app.testing import execution_compiled as ec
    assert cp.COMPOSITION_OPS == frozenset(ec._COMPOSERS.keys())


def test_validate_plan_does_not_raise_on_malformed_composition():
    # A plan-authoring bug (unrecognized op, missing composition) is NOT a hard validation error —
    # it degrades exactly like execution_compiled._compose does at runtime: silent fallback, never
    # a crash. See validate_plan's docstring.
    cp.validate_plan({
        "leaves": [{"id": "a", "instruction": "a"}],
        "agg_mode": "computed",
        "composition": {"op": "not_a_real_op"},
    })
    cp.validate_plan({
        "leaves": [{"id": "a", "instruction": "a"}],
        "agg_mode": "computed",
        # no composition key at all
    })


def test_validate_plan_preserves_agg_mode_and_composition_in_normalized_output():
    composition = {"op": "argmax", "items": [{"leaf": "a", "label": "A", "type": "number"}]}
    norm = cp.validate_plan({
        "leaves": [{"id": "a", "instruction": "a"}],
        "agg_mode": "computed",
        "composition": composition,
    })
    assert norm["agg_mode"] == "computed"
    assert norm["composition"] == composition
