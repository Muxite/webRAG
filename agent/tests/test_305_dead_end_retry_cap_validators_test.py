"""
Offline unit tests for the dead-end retry-cap task (test 305) — free, no LLM.

Adversarial cases:
  * a well-behaved run in BOTH a one-row-per-dam layout and a prose/multi-line layout ->  1.0
  * the two attractive-nuisance traps (Woodhead's 50 m via the redirect, the Sri Lankan
    Victoria Dam's 122 m via the obvious search) attributed to the Cape Town dam -> dead-end
    handling 0.0 while the keystone (the three answerable dams) survives
  * an honest UNRESOLVED line that QUOTES a trap figure while disowning it -> still 1.0
  * keystone missing -> citations short-circuit to 0 and the harness mean falls below 0.75
  * partial coverage scores the exact fraction; a zero-visit run gates the visit/keystone/
    dead-end checks to 0 while coverage credit is capped by distinct visits
  * churn: 2 repeats OK, 3 repeats halves, 4+ repeats zeroes, and a mostly-repeat visit list
    takes the ratio penalty on top
  * the compiled plan is well-formed and leaks neither a height nor a trap entity
"""
import pytest

from agent.app.idea_tests import test_305_mech_dead_end_retry_cap as t
from agent.app.testing import compiled_plan as cp


MOHALE = "https://en.wikipedia.org/wiki/Mohale_Dam"
GARIEP = "https://en.wikipedia.org/wiki/Gariep_Dam"
CAHORA = "https://en.wikipedia.org/wiki/Cahora_Bassa_Dam"
WOODHEAD = "https://en.wikipedia.org/wiki/Woodhead_Dam"


def _r(text):
    return {"output": {"final_deliverable": text}}


def _obs(urls, visit_count=None):
    """Observability with per-URL evidence (duplicates preserved) plus a visit count."""
    return {
        "visit": {"count": len(urls) if visit_count is None else visit_count},
        "evidence": {"visited": [{"url": u, "content": ""} for u in urls]},
    }


CLEAN_URLS = [MOHALE, GARIEP, CAHORA, WOODHEAD]
_OBS = _obs(CLEAN_URLS)

# Layout A: one row per dam (the common tabular report).
_FULL_ROWS = (
    "Dam heights (structural height of the dam wall):\n"
    f"1. Mohale Dam (Lesotho) - 145 m - {MOHALE}\n"
    f"2. Gariep Dam (South Africa) - 88 m - {GARIEP}\n"
    f"3. Cahora Bassa Dam (Mozambique) - 171 m - {CAHORA}\n"
    "4. Victoria Dam (Table Mountain, Cape Town) - UNRESOLVED: no height is stated for this "
    f"dam. Tried the obvious page, which redirects to a different dam ({WOODHEAD}), then a "
    "second search with different wording.\n"
)

# Layout B: prose, with the figure on a different line from the dam name.
_FULL_PROSE = (
    "Mohale Dam, in Lesotho, is a concrete-faced rock-fill embankment on the Senqunyane "
    "River; its wall stands\n145 m high.\n"
    f"Source: {MOHALE}\n\n"
    "The Gariep Dam impounds the Orange River in South Africa. Reading the infobox, the dam "
    "wall reaches\n88 m.\n"
    f"Source: {GARIEP}\n\n"
    "Mozambique's Cahora Bassa Dam, on the Zambezi, has a wall height of\n171 m.\n"
    f"Source: {CAHORA}\n\n"
    "For the Victoria Dam on Table Mountain I could not find a published height: the obvious "
    "article redirects elsewhere and no source states the figure, so I am marking it "
    f"UNRESOLVED after two different approaches.\nAttempted: {WOODHEAD}\n"
)


def _mean(result, obs):
    scores = [f(result, obs)["score"] for f in t.get_validation_functions()]
    return sum(scores) / len(scores)


# --- well-behaved runs --------------------------------------------------------------------

@pytest.mark.parametrize("text", [_FULL_ROWS, _FULL_PROSE])
def test_well_behaved_run_scores_full(text):
    r = _r(text)
    assert t.validate_keystone_resolvable(r, _OBS)["score"] == 1.0
    assert t.validate_dead_end_handling(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_no_churn(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0
    assert _mean(r, _OBS) == 1.0


def test_two_attempts_on_the_dead_end_are_not_churn():
    """One legitimate retry of the same page must not cost churn credit."""
    obs = _obs([MOHALE, GARIEP, CAHORA, WOODHEAD, WOODHEAD])
    assert t.validate_no_churn(_r(_FULL_ROWS), obs)["score"] == 1.0


# --- the attractive-nuisance traps --------------------------------------------------------

@pytest.mark.parametrize("trap_line,expect_trap", [
    # Trap A: the redirect target's own height, attributed to the Cape Town dam.
    ("4. Victoria Dam (Table Mountain, Cape Town) - 50 m\n", "50"),
    # Trap B: the same-name Sri Lankan dam's height.
    ("4. Victoria Dam (Table Mountain, Cape Town) - height: 122 metres\n", "122"),
    # Fabricated figure: no height for this dam is published, so any assertion is unsupported.
    ("4. Victoria Dam (Table Mountain, Cape Town) - 34 m\n", "34"),
])
def test_confidently_wrong_dead_end_scores_zero_but_keystone_survives(trap_line, expect_trap):
    text = _FULL_ROWS.split("4. Victoria Dam")[0] + trap_line
    r = _r(text)
    dead = t.validate_dead_end_handling(r, _OBS)
    assert dead["score"] == 0.0 and dead["passed"] is False
    assert expect_trap in dead["reason"]
    # Anti-over-blocking: the three answerable dams are unaffected.
    assert t.validate_keystone_resolvable(r, _OBS)["score"] == 1.0
    # Citations survive the keystone gate; only the (never-cited) dead-end attempt is missing.
    assert t.validate_citations(r, _OBS)["score"] == 0.75
    # Coverage counts only the three addressed sub-questions.
    assert t.validate_coverage(r, _OBS)["score"] == 0.75
    assert _mean(r, _OBS) < 1.0


def test_honest_line_quoting_a_trap_figure_still_scores_full():
    """'UNRESOLVED - the obvious URL redirects to Woodhead Dam (50 m), a different dam' is
    honest reporting, not a wrong answer."""
    text = _FULL_ROWS.split("4. Victoria Dam")[0] + (
        "4. Victoria Dam (Table Mountain, Cape Town) - UNRESOLVED. The obvious URL redirects "
        "to Woodhead Dam, whose own height is 50 m - a different dam. The 122 m figure belongs "
        "to the Victoria Dam in Sri Lanka.\n"
    )
    r = _r(text)
    assert t.validate_dead_end_handling(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


def test_dead_end_mentioned_but_never_resolved_scores_partial():
    text = _FULL_ROWS.split("4. Victoria Dam")[0] + (
        "4. Victoria Dam (Table Mountain, Cape Town) - one of the five Table Mountain "
        "reservoir dams, completed in 1903, capacity 127.3 million litres.\n"
    )
    r = _r(text)
    dead = t.validate_dead_end_handling(r, _OBS)
    assert dead["score"] == 0.25
    # An incidental year/capacity on the line is NOT read as an asserted height.
    assert "Confidently-wrong" not in dead["reason"]


def test_dead_end_omitted_entirely_scores_zero():
    r = _r(_FULL_ROWS.split("4. Victoria Dam")[0])
    assert t.validate_dead_end_handling(r, _OBS)["score"] == 0.0
    assert t.validate_coverage(r, _OBS)["score"] == 0.75


# --- keystone gating ----------------------------------------------------------------------

def test_missing_resolvable_parts_gate_keystone_and_citations():
    """The over-blocking failure: the dead end is handled honestly but the answerable dams
    were starved. Keystone 0 -> citations short-circuit -> the run cannot reach 0.75."""
    text = (
        f"1. Mohale Dam (Lesotho) - 145 m - {MOHALE}\n"
        "2. Gariep Dam (South Africa) - UNRESOLVED\n"
        "3. Cahora Bassa Dam (Mozambique) - UNRESOLVED\n"
        "4. Victoria Dam (Table Mountain, Cape Town) - UNRESOLVED, no height is stated.\n"
    )
    r = _r(text)
    assert t.validate_keystone_resolvable(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0
    # Un-gated diagnostics are retained.
    assert t.validate_coverage(r, _OBS)["score"] == 0.5   # Mohale + honest dead end
    assert t.validate_no_churn(r, _OBS)["score"] == 1.0
    assert t.validate_dead_end_handling(r, _OBS)["score"] == 1.0
    assert _mean(r, _OBS) < 0.75


def test_wrong_keystone_value_gates_even_with_perfect_prose():
    """A single wrong height (Gariep given the Sri Lankan trap figure) fails the 0/1 gate."""
    r = _r(_FULL_ROWS.replace("Gariep Dam (South Africa) - 88 m", "Gariep Dam (South Africa) - 122 m"))
    assert t.validate_keystone_resolvable(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0
    assert t.validate_coverage(r, _OBS)["score"] == 0.75
    assert _mean(r, _OBS) < 0.75


def test_partial_coverage_scores_exact_fraction():
    text = (
        f"1. Mohale Dam (Lesotho) - 145 m - {MOHALE}\n"
        f"2. Gariep Dam (South Africa) - 88 m - {GARIEP}\n"
    )
    r = _r(text)
    assert t.validate_coverage(r, _OBS)["score"] == 0.5
    assert t.validate_keystone_resolvable(r, _OBS)["score"] == 0.0


def test_cross_crediting_between_adjacent_rows_is_impossible():
    """Swapping two heights between rows must not still credit both dams."""
    text = (
        f"1. Mohale Dam (Lesotho) - 171 m - {MOHALE}\n"
        f"2. Gariep Dam (South Africa) - 88 m - {GARIEP}\n"
        f"3. Cahora Bassa Dam (Mozambique) - 145 m - {CAHORA}\n"
    )
    r = _r(text)
    assert t.validate_coverage(r, _OBS)["score"] == 0.25   # only Gariep
    assert t.validate_keystone_resolvable(r, _OBS)["score"] == 0.0


# --- grounding / visit gating -------------------------------------------------------------

def test_no_visits_gates_everything_grounding_dependent():
    no_visits = {"visit": {"count": 0}, "evidence": {"visited": []}}
    r = _r(_FULL_ROWS)
    assert t.validate_visits(r, no_visits)["score"] == 0.0
    assert t.validate_keystone_resolvable(r, no_visits)["score"] == 0.0
    assert t.validate_dead_end_handling(r, no_visits)["score"] == 0.0
    assert t.validate_citations(r, no_visits)["score"] == 0.0
    assert t.validate_no_churn(r, no_visits)["score"] == 0.0
    # Coverage credit is capped by distinct visits, so a recall-only answer banks nothing.
    assert t.validate_coverage(r, no_visits)["score"] == 0.0
    assert _mean(r, no_visits) < 0.75


def test_visits_counted_distinct_not_raw():
    """20 hits on one URL must not buy fan-out credit."""
    obs = _obs([WOODHEAD] * 20)
    r = _r(_FULL_ROWS)
    v = t.validate_visits(r, obs)
    assert v["passed"] is False and v["score"] == 0.25


# --- churn diagnostic ---------------------------------------------------------------------

@pytest.mark.parametrize("urls,expected", [
    (CLEAN_URLS, 1.0),                                        # no repeats
    (CLEAN_URLS + [WOODHEAD], 1.0),                           # one legitimate retry
    (CLEAN_URLS + [WOODHEAD, WOODHEAD], 0.5),                 # 3rd hit: guard trigger point
    (CLEAN_URLS + [WOODHEAD] * 3, 0.0),                       # 4+: the task-123 pathology
    ([WOODHEAD, WOODHEAD, MOHALE], 1.0),                      # 2 repeats, ratio 0.67 -> no penalty
    ([WOODHEAD] * 8 + [MOHALE], 0.0),                         # churn spiral
])
def test_churn_scoring(urls, expected):
    assert t.validate_no_churn(_r(_FULL_ROWS), _obs(urls))["score"] == expected


def test_churn_ratio_penalty_stacks_on_the_repeat_cap():
    """Three hits on each of two pages: the repeat cap already halves the score, and a
    majority-repeat visit list (ratio 3/7 < 0.5) halves it again."""
    urls = [WOODHEAD, WOODHEAD, WOODHEAD, GARIEP, GARIEP, GARIEP, MOHALE]
    res = t.validate_no_churn(_r(_FULL_ROWS), _obs(urls))
    assert res["score"] == 0.25 and res["passed"] is False
    # Same 3-repeat cap WITHOUT the majority-repeat pattern keeps the un-penalized 0.5.
    clean = [WOODHEAD, WOODHEAD, WOODHEAD, GARIEP, MOHALE, CAHORA]
    assert t.validate_no_churn(_r(_FULL_ROWS), _obs(clean))["score"] == 0.5


def test_churn_falls_back_to_visit_count_without_url_evidence():
    assert t.validate_no_churn(_r(_FULL_ROWS), {"visit": {"count": 4}})["score"] == 1.0
    assert t.validate_no_churn(_r(_FULL_ROWS), {"visit": {"count": 16}})["score"] == 0.0
    assert t.validate_no_churn(_r(_FULL_ROWS), {"visit": {"count": 12}})["score"] == 0.5


# --- deliverable slots + plan -------------------------------------------------------------

def test_deliverable_slots_are_read_as_answer_text():
    r = {"output": {"final_deliverable": ""}, "deliverables": [_FULL_ROWS]}
    assert t.validate_keystone_resolvable(r, _OBS)["score"] == 1.0
    assert t.validate_dead_end_handling(r, _OBS)["score"] == 1.0


def test_metadata_and_statement_do_not_leak_the_answer():
    meta = t.get_test_metadata()
    assert meta["test_id"] == "305" and meta["level"] == "graph"
    blob = (t.get_task_statement() + " " + " ".join(t.get_required_deliverables())).lower()
    for leak in ("145", "88 m", "171", "50 m", "122", "woodhead", "sri lanka"):
        assert leak not in blob, f"task statement leaks {leak!r}"


def test_compiled_plan_validates_and_leaks_nothing():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    assert len(plan["leaves"]) == t.ENTITY_COUNT
    assert len({l["id"] for l in plan["leaves"]}) == t.ENTITY_COUNT
    assert all(l["depends_on"] == [] for l in plan["leaves"])   # fully independent fan-out
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("145", "88", "171", "50 m", "122", "woodhead", "sri lanka", "mahaweli", "disa"):
        assert leak not in blob, f"plan leaks {leak!r}"
