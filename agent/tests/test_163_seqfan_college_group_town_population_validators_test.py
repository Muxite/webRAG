"""
Offline unit tests for the tier-5 sequential-prefix -> 7-way fan-out -> argmin task (test 163).
Free: no LLM, no network.

Adversarial cases: a grounded full answer in BOTH a single-line-per-row table layout and a
multi-line prose layout (each must reach 1.0 on every check), a wrong argmin picked from the
runner-up branch (keystone 0, gated secondaries 0, UN-gated coverage retained in full), a
partially-resolved roster at an exact fraction, a fabricated no-visit answer (grounding gate),
superlative false-triggers that must NOT open the gate, the visit diagnostic, the ground-truth
margin invariant, and the compiled plan being well-formed, correctly staged and leak-free.
"""
import re

from agent.app.idea_tests import test_163_tier5_seqfan_college_group_town_population as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 17}}
_NO_VISITS = {"visit": {"count": 0}}

_ROWS_SINGLE_LINE = (
    "Bryn Mawr College - Bryn Mawr, Pennsylvania - 5,879 - "
    "https://en.wikipedia.org/wiki/Bryn_Mawr,_Pennsylvania\n"
    "Mount Holyoke College - South Hadley, Massachusetts - 18,150 - "
    "https://en.wikipedia.org/wiki/South_Hadley,_Massachusetts\n"
    "Wellesley College - Wellesley, Massachusetts - 29,550 - "
    "https://en.wikipedia.org/wiki/Wellesley,_Massachusetts\n"
    "Smith College - Northampton, Massachusetts - 29,571 - "
    "https://en.wikipedia.org/wiki/Northampton,_Massachusetts\n"
    "Vassar College - Poughkeepsie, New York - 31,577 - "
    "https://en.wikipedia.org/wiki/Poughkeepsie,_New_York\n"
    "Radcliffe College - Cambridge, Massachusetts - 118,403 - "
    "https://en.wikipedia.org/wiki/Cambridge,_Massachusetts\n"
    "Barnard College - Manhattan, New York - 1,694,251 - "
    "https://en.wikipedia.org/wiki/Manhattan\n"
)

_FULL_TABLE = (
    "Prefix: Matthew Vassar founded Vassar College, one of the Seven Sisters colleges.\n"
    "Answer: Bryn Mawr College is in the least populous municipality, Bryn Mawr, "
    "Pennsylvania (population 5,879).\n\n" + _ROWS_SINGLE_LINE
)

# Multi-line layout: the verdict label and the college name are separated by a newline, which a
# [^.\n] proximity window would wrongly reject.
_FULL_PROSE = (
    "Chain: Matthew Vassar -> Vassar College -> the Seven Sisters colleges.\n\n"
    "Least populous municipality:\n"
    "Bryn Mawr College\n"
    "(Bryn Mawr, Pennsylvania, population 5879)\n\n"
    "Roster:\n"
    "Mount Holyoke College, in South Hadley, Massachusetts, has 18150 residents "
    "(https://en_wikipedia_org/wiki/South_Hadley,_Massachusetts)\n"
    "Wellesley College, in Wellesley, Massachusetts, has 29550 residents "
    "(https://en_wikipedia_org/wiki/Wellesley,_Massachusetts)\n"
    "Smith College, in Northampton, Massachusetts, has 29571 residents "
    "(https://en_wikipedia_org/wiki/Northampton,_Massachusetts)\n"
    "Vassar College, in Poughkeepsie, New York, has 31577 residents "
    "(https://en_wikipedia_org/wiki/Poughkeepsie,_New_York)\n"
    "Radcliffe College, in Cambridge, Massachusetts, has 118403 residents "
    "(https://en_wikipedia_org/wiki/Cambridge,_Massachusetts)\n"
    "Barnard College, in Manhattan, New York, has 1694251 residents "
    "(https://en_wikipedia_org/wiki/Manhattan)\n"
    "Bryn Mawr College, in Bryn Mawr, Pennsylvania, has 5879 residents "
    "(https://en_wikipedia_org/wiki/Bryn_Mawr,_Pennsylvania)\n"
)


def _by_check(result, obs):
    return {v(result, obs)["check"]: v(result, obs) for v in t.get_validation_functions()}


def test_full_answer_single_line_layout_scores_one_everywhere():
    checks = _by_check(_r(_FULL_TABLE), _OBS)
    for name in ("visit_count", "keystone_least_populous", "coverage", "prefix_chain", "citations"):
        assert checks[name]["passed"], (name, checks[name]["reason"])
        assert checks[name]["score"] == 1.0, (name, checks[name])


def test_full_answer_multi_line_layout_scores_one_everywhere():
    checks = _by_check(_r(_FULL_PROSE), _OBS)
    for name in ("visit_count", "keystone_least_populous", "coverage", "prefix_chain", "citations"):
        assert checks[name]["passed"], (name, checks[name]["reason"])
        assert checks[name]["score"] == 1.0, (name, checks[name])


def test_wrong_argmin_gates_keystone_and_secondaries_but_keeps_coverage():
    text = _FULL_TABLE.replace(
        "Bryn Mawr College is in the least populous municipality, Bryn Mawr, "
        "Pennsylvania (population 5,879).",
        "Mount Holyoke College is in the least populous municipality, South Hadley, "
        "Massachusetts (population 18,150).",
    )
    checks = _by_check(_r(text), _OBS)
    assert checks["keystone_least_populous"]["score"] == 0.0
    assert checks["prefix_chain"]["score"] == 0.0
    assert checks["citations"]["score"] == 0.0
    assert checks["coverage"]["score"] == 1.0, checks["coverage"]["reason"]


def test_partial_roster_scores_exact_fraction_and_fails_keystone():
    kept = _ROWS_SINGLE_LINE.splitlines(keepends=True)[1:4]
    checks = _by_check(_r("Roster so far:\n" + "".join(kept)), _OBS)
    assert checks["coverage"]["score"] == 3 / 7
    assert not checks["coverage"]["passed"]
    assert checks["keystone_least_populous"]["score"] == 0.0


def test_fabricated_answer_without_visits_scores_zero_on_gated_checks():
    checks = _by_check(_r(_FULL_TABLE), _NO_VISITS)
    assert checks["keystone_least_populous"]["score"] == 0.0
    assert checks["prefix_chain"]["score"] == 0.0
    assert checks["citations"]["score"] == 0.0
    assert checks["visit_count"]["score"] == 0.0
    # The UN-gated breadth diagnostic still reports what the text claimed.
    assert checks["coverage"]["score"] == 1.0


def test_keystone_needs_both_the_name_and_the_population():
    no_pop = "Bryn Mawr College sits in the least populous municipality of the seven."
    assert t.validate_keystone_least_populous(_r(no_pop), _OBS)["score"] == 0.0
    no_name = "The least populous municipality on the roster has a population of 5,879."
    assert t.validate_keystone_least_populous(_r(no_name), _OBS)["score"] == 0.0


def test_non_superlative_mentions_do_not_open_the_gate():
    for text in (
        "Bryn Mawr, Pennsylvania is a small community of 5,879 people.",
        "Bryn Mawr (5,879) is less populated than South Hadley.",
        "Bryn Mawr College, 5,879. Mount Holyoke College is in the least populous county.",
    ):
        assert t.validate_keystone_least_populous(_r(text), _OBS)["score"] == 0.0, text


def test_visit_gate_thresholds():
    assert not t.validate_visits(_r(""), {"visit": {"count": 5}})["passed"]
    assert t.validate_visits(_r(""), {"visit": {"count": 6}})["passed"]
    assert t.validate_visits(_r(""), {"visit": {"count": 17}})["score"] == 1.0


def test_ground_truth_margin_is_wide():
    pops = sorted(m["population"] for m in t.MEMBERS)
    assert t.KEYSTONE["college"] == "Bryn Mawr College"
    assert pops[0] == 5879 and pops[1] == 18150
    assert pops[1] - pops[0] > 10000
    assert pops[1] / pops[0] > 3.0
    assert len(t.MEMBERS) == 7


def test_task_statement_does_not_name_the_roster_or_the_answer():
    statement = t.get_task_statement().lower()
    assert "matthew vassar" in statement
    assert "seven sisters" not in statement
    for m in t.MEMBERS:
        assert m["college"].lower() not in statement, m["college"]
        assert m["town"].lower() not in statement, m["town"]
        assert str(m["population"]) not in statement


def test_compiled_plan_is_well_formed_and_correctly_staged():
    plan = cp.validate_plan(t.get_compiled_plan())
    ids = [leaf["id"] for leaf in plan["leaves"]]
    assert ids[:3] == ["founded_college", "historic_group", "roster"]
    assert len(plan["leaves"]) == 3 + len(t.MEMBERS)
    waves = cp.topological_waves(plan["leaves"])
    assert [len(w) for w in waves] == [1, 1, 1, len(t.MEMBERS)]
    branch = [leaf for leaf in plan["leaves"] if leaf["id"].startswith("member_")]
    assert len(branch) == len(t.MEMBERS)
    for leaf in branch:
        assert leaf["depends_on"] == ["roster"], leaf["id"]
        assert "{roster}" in leaf["instruction"]


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(
        [str(leaf.get("instruction", "")) + " " + str(leaf.get("expect", ""))
         for leaf in plan["leaves"]] + [str(plan["aggregation"])]
    )
    # The GIVEN start person is allowed; strip it so its surname cannot mask a real leak.
    blob = blob.replace(t.START_PERSON, "<start>")
    lowered = blob.lower()
    for member in t.MEMBERS:
        for token in (member["college"], member["town"], member["town"].split(",")[0],
                      str(member["population"]), f"{member['population']:,}"):
            assert token.lower() not in lowered, f"plan leaks {token!r}"
    # "least populous" is the merge OPERATION and legitimately appears in the aggregation
    # recipe; what must never appear is any entity, municipality or figure that answers it.
    for token in ("seven sisters", "bryn mawr", "pennsylvania", "5,879", "5879"):
        assert token not in lowered, f"plan leaks {token!r}"
    for waypoint in t.PREFIX_WAYPOINTS:
        assert not re.search(waypoint["name_rx"], lowered), waypoint["name"]
    assert t.validate_keystone_least_populous(_r(blob), _OBS)["score"] == 0.0
