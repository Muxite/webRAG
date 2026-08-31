"""
Offline unit tests for the N-sweep task family (tests 165 / 166 / 167 / 168).
Free: no LLM, no network.

Every case runs against ALL FOUR family members, because the point of the family is that only
N differs -- a check that holds at N=4 but not at N=32 is a defect in the template, not in the
answer. Adversarial cases: a grounded full answer in BOTH a single-line table layout and a
multi-line prose layout (1.0 everywhere), a wrong argmin (keystone 0, gated secondaries 0,
un-gated recall retained), partial coverage at an exact k/N, a fabricated no-visit answer, the
decision-critical item dropped, early-band eviction (the mechanism signature), superlative
false-triggers, the ground-truth margin and nesting invariants, and the compiled plan being
well-formed, correctly staged and leak-free.
"""
import re

import pytest

from agent.app.idea_tests import nsweep_toho_runtime_family as family
from agent.app.idea_tests import test_165_tier5_nsweep_toho_runtime_n04 as t165
from agent.app.idea_tests import test_166_tier5_nsweep_toho_runtime_n08 as t166
from agent.app.idea_tests import test_167_tier5_nsweep_toho_runtime_n16 as t167
from agent.app.idea_tests import test_168_tier5_nsweep_toho_runtime_n32 as t168
from agent.app.testing import compiled_plan as cp


MODULES = [t165, t166, t167, t168]
CHECKS = ("visit_count", "keystone_shortest_film", "coverage", "positional_recall",
          "total_runtime", "citations")


def _r(text):
    return {"output": {"final_deliverable": text}}


def _obs(mod):
    return {"visit": {"count": mod.N + 1}}


_NO_VISITS = {"visit": {"count": 0}}


def _table_rows(mod, indices=None):
    """One line per film: number - title - runtime - URL."""
    keep = set(indices if indices is not None else [f["index"] for f in mod.ROSTER])
    return "".join(
        f"{f['index']}. {f['title']} - {f['runtime']} minutes - {f['url']}\n"
        for f in mod.ROSTER if f["index"] in keep
    )


def _prose_rows(mod, indices=None):
    """Multi-line prose layout: the figure sits a line below the title."""
    keep = set(indices if indices is not None else [f["index"] for f in mod.ROSTER])
    return "".join(
        f"{f['title']}\n  runs {f['runtime']} minutes\n  source: {f['url']}\n"
        for f in mod.ROSTER if f["index"] in keep
    )


def _verdict(mod, film=None):
    film = film or mod.KEYSTONE
    return (f"Shortest running time: {film['title']} at {film['runtime']} minutes\n"
            f"Combined total running time of the {mod.N} films: {mod.TOTAL_RUNTIME} minutes\n\n")


def _full_table(mod):
    return _verdict(mod) + _table_rows(mod)


def _full_prose(mod):
    return (f"The shortest of the {mod.N}:\n{mod.KEYSTONE['title']}\n"
            f"({mod.KEYSTONE['runtime']} minutes)\n\n"
            f"Total combined running time: {mod.TOTAL_RUNTIME} minutes\n\n" + _prose_rows(mod))


def _by_check(mod, result, obs):
    return {v(result, obs)["check"]: v(result, obs) for v in mod.get_validation_functions()}


@pytest.mark.parametrize("mod", MODULES)
def test_full_answer_single_line_layout_scores_one_everywhere(mod):
    checks = _by_check(mod, _r(_full_table(mod)), _obs(mod))
    for name in CHECKS:
        assert checks[name]["passed"], (mod.N, name, checks[name]["reason"])
        assert checks[name]["score"] == 1.0, (mod.N, name, checks[name])


@pytest.mark.parametrize("mod", MODULES)
def test_full_answer_multi_line_layout_scores_one_everywhere(mod):
    checks = _by_check(mod, _r(_full_prose(mod)), _obs(mod))
    for name in CHECKS:
        assert checks[name]["passed"], (mod.N, name, checks[name]["reason"])
        assert checks[name]["score"] == 1.0, (mod.N, name, checks[name])


@pytest.mark.parametrize("mod", MODULES)
def test_wrong_argmin_gates_keystone_and_secondaries_but_keeps_recall(mod):
    runner_up = min((f for f in mod.ROSTER if f["index"] != mod.KEYSTONE["index"]),
                    key=lambda f: f["runtime"])
    checks = _by_check(mod, _r(_verdict(mod, runner_up) + _table_rows(mod)), _obs(mod))
    assert checks["keystone_shortest_film"]["score"] == 0.0
    assert checks["total_runtime"]["score"] == 0.0
    assert checks["citations"]["score"] == 0.0
    assert checks["coverage"]["score"] == 1.0, checks["coverage"]["reason"]
    assert checks["positional_recall"]["score"] == 1.0


@pytest.mark.parametrize("mod", MODULES)
def test_partial_coverage_reports_the_exact_fraction(mod):
    kept = [f["index"] for f in mod.ROSTER][: max(1, mod.N // 4)]
    checks = _by_check(mod, _r("Partial roster:\n" + _table_rows(mod, kept)), _obs(mod))
    assert checks["coverage"]["score"] == len(kept) / float(mod.N)
    assert checks["coverage"]["detail"]["resolved"] == kept
    assert not checks["coverage"]["passed"]
    assert checks["keystone_shortest_film"]["score"] == 0.0


@pytest.mark.parametrize("mod", MODULES)
def test_fabricated_answer_without_visits_scores_zero_on_gated_checks(mod):
    checks = _by_check(mod, _r(_full_table(mod)), _NO_VISITS)
    assert checks["visit_count"]["score"] == 0.0
    assert not checks["visit_count"]["passed"]
    assert checks["keystone_shortest_film"]["score"] == 0.0
    assert checks["total_runtime"]["score"] == 0.0
    assert checks["citations"]["score"] == 0.0
    # The un-gated recall diagnostics still report what the text claimed.
    assert checks["coverage"]["score"] == 1.0


@pytest.mark.parametrize("mod", MODULES)
def test_missing_decision_critical_item_breaks_the_keystone(mod):
    survivors = [f["index"] for f in mod.ROSTER if f["index"] != mod.KEYSTONE["index"]]
    runner_up = min((f for f in mod.ROSTER if f["index"] in survivors),
                    key=lambda f: f["runtime"])
    text = _verdict(mod, runner_up) + _table_rows(mod, survivors)
    checks = _by_check(mod, _r(text), _obs(mod))
    assert checks["keystone_shortest_film"]["score"] == 0.0
    assert checks["coverage"]["score"] == (mod.N - 1) / float(mod.N)
    assert mod.KEYSTONE["index"] not in checks["coverage"]["detail"]["resolved"]


@pytest.mark.parametrize("mod", MODULES)
def test_early_band_eviction_is_visible_in_positional_recall(mod):
    early, middle, late = family.segments(mod.N)
    kept = middle + late
    checks = _by_check(mod, _r("Roster:\n" + _table_rows(mod, kept)), _obs(mod))
    assert checks["positional_recall"]["score"] == 0.0
    assert checks["positional_recall"]["detail"]["band_scores"] == [0.0, 1.0, 1.0]
    assert checks["coverage"]["score"] == len(kept) / float(mod.N)


@pytest.mark.parametrize("mod", MODULES)
def test_keystone_needs_both_the_title_and_the_runtime(mod):
    no_value = f"{mod.KEYSTONE['title']} has the shortest running time of the {mod.N}."
    assert mod.get_validation_functions()[1](_r(no_value), _obs(mod))["score"] == 0.0
    no_title = f"The shortest running time on the roster is {mod.KEYSTONE['runtime']} minutes."
    assert mod.get_validation_functions()[1](_r(no_title), _obs(mod))["score"] == 0.0


@pytest.mark.parametrize("mod", MODULES)
def test_non_superlative_mentions_do_not_open_the_gate(mod):
    k = mod.KEYSTONE
    for text in (
        f"{k['title']} is a short film running {k['runtime']} minutes.",
        f"{k['title']} ({k['runtime']} min) is shorter than the others.",
        f"{k['title']}: {k['runtime']} minutes. Another film has the shortest running time.",
    ):
        assert mod.get_validation_functions()[1](_r(text), _obs(mod))["score"] == 0.0, text


@pytest.mark.parametrize("mod", MODULES)
def test_visit_gate_thresholds(mod):
    validate_visits = mod.get_validation_functions()[0]
    assert validate_visits(_r(""), {"visit": {"count": 0}})["score"] == 0.0
    assert not validate_visits(_r(""), {"visit": {"count": 1}})["passed"]
    assert validate_visits(_r(""), {"visit": {"count": 2}})["passed"]
    assert validate_visits(_r(""), {"visit": {"count": mod.N + 1}})["score"] == 1.0


@pytest.mark.parametrize("mod", MODULES)
def test_total_runtime_is_gated_and_exact(mod):
    validate_total = mod.get_validation_functions()[4]
    off_by_one = _verdict(mod).replace(f"{mod.TOTAL_RUNTIME} minutes",
                                       f"{mod.TOTAL_RUNTIME - 1} minutes")
    assert validate_total(_r(off_by_one + _table_rows(mod)), _obs(mod))["score"] == 0.0
    assert validate_total(_r(_full_table(mod)), _obs(mod))["score"] == 1.0


def test_family_is_a_nested_prefix_with_flat_per_item_difficulty():
    rosters = [family.roster(n) for n in family.SWEEP_SIZES]
    for smaller, bigger in zip(rosters, rosters[1:]):
        assert bigger[: len(smaller)] == smaller
    # The mandate, deliverables and plan differ from each other ONLY by the number N.
    for n in family.SWEEP_SIZES:
        normalized = re.sub(r"\b(?:4|8|16|32)\b", "<N>", family.get_task_statement(n))
        assert normalized == re.sub(r"\b(?:4|8|16|32)\b", "<N>",
                                    family.get_task_statement(family.SWEEP_SIZES[0]))


def test_ground_truth_margins_and_totals():
    assert [family.keystone(n)["title"] for n in family.SWEEP_SIZES] == [
        "Godzilla Raids Again", "Godzilla Raids Again",
        "All Monsters Attack", "All Monsters Attack"]
    assert [family.keystone(n)["runtime"] for n in family.SWEEP_SIZES] == [81, 81, 70, 70]
    assert [family.margin(n) for n in family.SWEEP_SIZES] == [7, 5, 11, 11]
    assert [family.total_runtime(n) for n in family.SWEEP_SIZES] == [362, 722, 1405, 3051]
    for n in family.SWEEP_SIZES:
        runtimes = sorted(f["runtime"] for f in family.roster(n))
        # the argmin must be strictly unique, or "which film" has no single answer
        assert runtimes[0] < runtimes[1]
        # no plausible single misread of another item can undercut the winner
        assert family.margin(n) >= 5
    assert len(family.FILMS) == 32
    assert len({f["index"] for f in family.FILMS}) == 32


def test_title_regexes_are_mutually_exclusive():
    for film in family.FILMS:
        matches = [other["index"] for other in family.FILMS
                   if re.search(film["title_rx"], other["title"].lower())]
        assert matches == [film["index"]], (film["title"], matches)


@pytest.mark.parametrize("mod", MODULES)
def test_task_statement_leaks_no_title_runtime_or_answer(mod):
    statement = mod.get_task_statement().lower()
    assert family.ROSTER_PAGE.lower() in statement
    for film in family.FILMS:
        for token in film["leak_tokens"]:
            assert token not in statement, (mod.N, token)
        assert not re.search(film["title_rx"], statement), (mod.N, film["title"])
        assert not re.search(film["value_rx"], statement), (mod.N, film["runtime"])
    assert str(mod.TOTAL_RUNTIME) not in statement


@pytest.mark.parametrize("mod", MODULES)
def test_compiled_plan_is_well_formed_and_correctly_staged(mod):
    plan = cp.validate_plan(mod.get_compiled_plan())
    ids = [leaf["id"] for leaf in plan["leaves"]]
    assert ids[0] == "roster"
    assert ids[1:] == [f"film_{i}" for i in range(1, mod.N + 1)]
    waves = cp.topological_waves(plan["leaves"])
    assert [len(w) for w in waves] == [1, mod.N]
    for leaf in plan["leaves"][1:]:
        assert leaf["depends_on"] == ["roster"], leaf["id"]
        assert "{roster}" in leaf["instruction"]


@pytest.mark.parametrize("mod", MODULES)
def test_compiled_plan_leaks_nothing(mod):
    plan = mod.get_compiled_plan()
    blob = " ".join([str(leaf.get("instruction", "")) + " " + str(leaf.get("expect", ""))
                     for leaf in plan["leaves"]] + [str(plan["aggregation"])])
    lowered = blob.lower()
    # The GIVEN roster page is allowed; strip it so its words cannot mask a real leak.
    lowered = lowered.replace(family.ROSTER_PAGE.lower(), "<roster page>")
    for film in family.FILMS:
        for token in film["leak_tokens"]:
            assert token not in lowered, (mod.N, token)
        assert not re.search(film["title_rx"], lowered), (mod.N, film["title"])
        assert not re.search(film["value_rx"], lowered), (mod.N, film["runtime"])
    assert str(mod.TOTAL_RUNTIME) not in lowered
    # "shortest" is the merge OPERATION and legitimately appears in the aggregation recipe;
    # what must never appear is the film that answers it.
    assert mod.get_validation_functions()[1](_r(blob), _obs(mod))["score"] == 0.0


@pytest.mark.parametrize("mod", MODULES)
def test_metadata_is_template_identical_except_for_n(mod):
    meta = mod.get_test_metadata()
    assert meta["level"] == "graph"
    assert meta["weight"] == "long"
    assert meta["category"] == "N-sweep Breadth Aggregation"
    assert meta["test_id"] in ("165", "166", "167", "168")
    assert mod.get_llm_validation_function() is None
    assert len(mod.get_validation_functions()) == len(CHECKS)


@pytest.mark.parametrize("mod", MODULES)
def test_shifted_values_earn_credit_only_where_the_text_is_actually_right(mod):
    """Every figure moved one row up: the correct figure still sits ~12 characters from each
    title, so a naive proximity window would hand out near-full credit. Nearest-preceding-title
    attribution must reject every row except the ones where the shift happens to reprint a
    film's true runtime (12 of the 32 runtimes are shared by another film)."""
    rows = mod.ROSTER
    shifted = {f["index"]: rows[(i + 1) % len(rows)]["runtime"] for i, f in enumerate(rows)}
    text = "".join(f"{f['index']}. {f['title']} - {shifted[f['index']]} minutes\n" for f in rows)
    credited = family.resolved_indices(text, mod.N)
    for index in credited:
        film = next(f for f in rows if f["index"] == index)
        assert shifted[index] == film["runtime"], (mod.N, index)
    assert len(credited) < mod.N
