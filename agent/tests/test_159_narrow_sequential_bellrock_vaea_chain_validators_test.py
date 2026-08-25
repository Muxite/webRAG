"""
Offline unit tests for the narrow-sequential 4-hop chain (test 159) — free, no LLM.

Adversarial coverage: full answer in single- AND multi-line layout (1.0 across the board), an
ungrounded parametric-memory answer (visit gate -> keystone/hop-resolution/citations all 0), the
two realistic wrong-answer modes for this chain (STOP-EARLY at the lighthouse's own height,
WRONG-BRANCH to the island nation's highest peak), exact partial-coverage fractions, per-waypoint
evidence grounding, the un-gated path-efficiency cost diagnostic that prices needless fan-out on a
width-1 chain, and the compiled plan being a well-formed 4-leaf DAG chain that leaks nothing.
"""
from agent.app.idea_tests import test_159_tier5_narrow_sequential_bellrock_vaea_chain as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


# Per-waypoint visited-page evidence (the grounding channel validate_chain_coverage checks).
_EV_START = {"url": "https://en.wikipedia.org/wiki/Bell_Rock_Lighthouse",
             "content": "The Bell Rock Lighthouse off Angus was built under the direction of "
                        "the engineer who published the account of the works."}
_EV_ENGINEER = {"url": "https://en.wikipedia.org/wiki/Robert_Stevenson_(civil_engineer)",
                "content": "Robert Stevenson was a Scottish civil engineer; his son Thomas was "
                           "the father of the author Robert Louis Stevenson."}
_EV_NOVELIST = {"url": "https://en.wikipedia.org/wiki/Robert_Louis_Stevenson",
                "content": "Resting place: Mount Vaea. They bore him to nearby Mount Vaea, where "
                           "they buried him on a spot overlooking the sea."}
_EV_TERMINAL = {"url": "https://en.wikipedia.org/wiki/Mount_Vaea",
                "content": "Mount Vaea is a mountain on Upolu, Samoa. Elevation 472 m (1,549 ft)."}
_FULL_EVIDENCE = [_EV_START, _EV_ENGINEER, _EV_NOVELIST, _EV_TERMINAL]


def _obs(visited=None, n=4):
    return {"visit": {"count": n},
            "evidence": {"visited": _FULL_EVIDENCE if visited is None else visited}}


_OBS = _obs()

_FULL_SINGLE = (
    "Hop 1: the Bell Rock Lighthouse "
    "(https://en.wikipedia.org/wiki/Bell_Rock_Lighthouse) was directed by Robert Stevenson "
    "(https://en.wikipedia.org/wiki/Robert_Stevenson_(civil_engineer)). Hop 2: his grandson was the "
    "novelist Robert Louis Stevenson "
    "(https://en.wikipedia.org/wiki/Robert_Louis_Stevenson). Hop 3: he is buried on Mount Vaea "
    "(https://en.wikipedia.org/wiki/Mount_Vaea). Hop 4: its elevation is 472 m (1,549 ft)."
)

_FULL_MULTI = (
    "HOP 1 — engineer:\n"
    "  Bell Rock Lighthouse -> Robert Stevenson\n"
    "    https://en.wikipedia.org/wiki/Bell_Rock_Lighthouse\n"
    "    https://en.wikipedia.org/wiki/Robert_Stevenson_(civil_engineer)\n"
    "HOP 2 — novelist grandson:\n"
    "  Robert Louis Stevenson\n"
    "    https://en.wikipedia.org/wiki/Robert_Louis_Stevenson\n"
    "HOP 3 — resting place:\n"
    "  Mount Vaea\n"
    "    https://en.wikipedia.org/wiki/Mount_Vaea\n"
    "HOP 4 — elevation:\n"
    "  472 m\n"
    "  (1,549 ft)\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_elevation(r, _OBS)["score"] == 1.0
    assert t.validate_chain_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_path_efficiency(r, _OBS)["score"] == 1.0
    assert t.validate_hop_resolution(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0
    scores = [f(r, _OBS)["score"] for f in t.get_validation_functions()]
    assert sum(scores) / len(scores) == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_elevation(r, _OBS)["score"] == 1.0
    assert t.validate_chain_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_hop_resolution(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True


def test_ungrounded_correct_value_gates_to_zero():
    """The correct keystone VALUE alone must not earn credit with zero visits — an ungrounded
    parametric-memory answer collapses the whole mean well below the 0.75 bar."""
    r = _r(_FULL_SINGLE)
    ungrounded = {"visit": {"count": 0}}
    assert t.validate_keystone_elevation(r, ungrounded)["passed"] is False
    assert t.validate_keystone_elevation(r, ungrounded)["score"] == 0.0
    assert t.validate_hop_resolution(r, ungrounded)["score"] == 0.0
    assert t.validate_citations(r, ungrounded)["score"] == 0.0
    assert t.validate_chain_coverage(r, ungrounded)["score"] == 0.0
    assert t.validate_path_efficiency(r, ungrounded)["score"] == 0.0
    scores = [f(r, ungrounded)["score"] for f in t.get_validation_functions()]
    assert sum(scores) / len(scores) < 0.75


def test_wrong_branch_highest_peak_gates_to_zero_but_keeps_coverage():
    """Hop-3 failure mode: guessing the island nation's highest peak (Mount Silisili, 1,858 m)
    instead of reading the burial mountain off the novelist's page."""
    wrong = (
        "Robert Louis Stevenson, grandson of the Bell Rock Lighthouse engineer Robert Stevenson, "
        "lived in Samoa; the country's highest peak is Mount Silisili at 1,858 m (6,096 ft)."
    )
    r = _r(wrong)
    partial = _obs([_EV_START, _EV_ENGINEER, _EV_NOVELIST])  # terminal page never opened
    assert t.validate_keystone_elevation(r, partial)["score"] == 0.0
    assert t.validate_hop_resolution(r, partial)["score"] == 0.0
    assert t.validate_citations(r, partial)["score"] == 0.0
    # Depth diagnostic is un-gated: three of four waypoints were genuinely traversed.
    assert abs(t.validate_chain_coverage(r, partial)["score"] - 3 / 4) < 1e-9


def test_stop_early_lighthouse_height_gates_to_zero():
    wrong = ("The Bell Rock Lighthouse, built by Robert Stevenson, has a tower height of "
             "36 m (118 ft).")
    r = _r(wrong)
    partial = _obs([_EV_START, _EV_ENGINEER], n=2)
    assert t.validate_keystone_elevation(r, partial)["score"] == 0.0
    assert t.validate_hop_resolution(r, partial)["score"] == 0.0
    assert t.validate_citations(r, partial)["score"] == 0.0
    assert abs(t.validate_chain_coverage(r, partial)["score"] - 2 / 4) < 1e-9
    assert t.validate_visits(r, partial)["passed"] is False


def test_keystone_token_rejects_embedded_and_near_miss():
    assert t.validate_keystone_elevation(_r("serial 4720 xj"), _OBS)["score"] == 0.0
    assert t.validate_keystone_elevation(_r("marker 15490"), _OBS)["score"] == 0.0
    assert t.validate_keystone_elevation(_r("ratio 4.72 units"), _OBS)["score"] == 0.0
    assert t.validate_keystone_elevation(_r("height 471 m"), _OBS)["score"] == 0.0
    # Both accepted unit spellings still pass.
    assert t.validate_keystone_elevation(_r("elevation 1,549 ft"), _OBS)["score"] == 1.0
    assert t.validate_keystone_elevation(_r("elevation 1549 ft"), _OBS)["score"] == 1.0


def test_partial_coverage_scores_exact_fraction():
    text = "I read the Bell Rock Lighthouse page and found Robert Stevenson; I got no further."
    r = _r(text)
    partial = _obs([_EV_START, _EV_ENGINEER], n=2)
    assert abs(t.validate_chain_coverage(r, partial)["score"] - 2 / 4) < 1e-9
    assert t.validate_keystone_elevation(r, partial)["score"] == 0.0


def test_chain_coverage_requires_page_evidence_not_just_text():
    """A waypoint named in the answer with no supporting visited-page evidence earns no credit,
    no matter how many other pages were visited."""
    r = _r(_FULL_SINGLE)
    assert t.validate_chain_coverage(r, _obs([]))["score"] == 0.0
    # Only the start page opened: it supports "start" alone (its content names no downstream hop).
    assert abs(t.validate_chain_coverage(r, _obs([_EV_START]))["score"] - 1 / 4) < 1e-9
    assert t.validate_chain_coverage(r, _obs(_FULL_EVIDENCE))["score"] == 1.0


def test_engineer_and_novelist_waypoints_are_separately_matched():
    """The hop-1 engineer regex must not be satisfied by the hop-2 novelist's name (otherwise a
    single-hop answer would score two waypoints)."""
    only_novelist = _r("Robert Louis Stevenson is buried on Mount Vaea.")
    assert abs(t.validate_chain_coverage(only_novelist, _OBS)["score"] - 2 / 4) < 1e-9


def test_path_efficiency_prices_needless_fanout_but_does_not_fail_a_correct_run():
    """The mechanism this task exists for: a width-1 chain rewards no speculative branching. A
    breadth-oriented engine that opens many off-path pages decays on this cost axis while a
    golden-path linear run scores 1.0 — and a correct-but-wasteful run still clears the 0.75 bar,
    so the diagnostic prices overhead instead of failing it."""
    r = _r(_FULL_SINGLE)
    assert t.validate_path_efficiency(r, _obs(n=4))["score"] == 1.0
    assert t.validate_path_efficiency(r, _obs(n=2))["score"] == 1.0  # fewer than ideal: not punished
    assert abs(t.validate_path_efficiency(r, _obs(n=8))["score"] - 0.5) < 1e-9
    assert abs(t.validate_path_efficiency(r, _obs(n=16))["score"] - 0.25) < 1e-9
    assert t.validate_path_efficiency(r, _obs(n=8))["passed"] is False
    assert t.validate_path_efficiency(r, _obs(n=6))["passed"] is True
    wasteful = _obs(n=16)
    scores = [f(r, wasteful)["score"] for f in t.get_validation_functions()]
    assert sum(scores) / len(scores) >= 0.75


def test_visit_count_fallback_to_evidence_when_counter_missing():
    r = _r(_FULL_SINGLE)
    obs = {"evidence": {"visited": _FULL_EVIDENCE}}
    assert t.validate_visits(r, obs)["score"] == 1.0
    assert t.validate_keystone_elevation(r, obs)["score"] == 1.0


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["Mount Vaea elevation: 472 m (1,549 ft)", "novelist: Robert Louis Stevenson"]}
    assert t.validate_keystone_elevation(r, _OBS)["score"] == 1.0


def test_compiled_plan_validates_and_is_dag_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 4
    assert struct["edge_count"] == 3
    assert struct["wave_widths"] == [1, 1, 1, 1]
    assert struct["is_dag_chain"] is True
    assert struct["is_pure_fanout"] is False


def test_compiled_plan_templates_upstream_and_leaks_nothing():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    assert "{engineer}" in by_id["novelist"]["instruction"]
    assert "{novelist}" in by_id["resting_place"]["instruction"]
    assert "{resting_place}" in by_id["elevation"]["instruction"]
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("stevenson", "vaea", "472", "1,549", "1549", "samoa", "silisili", "upolu"):
        assert leak not in blob, f"plan leaks {leak!r}"


def test_task_statement_leaks_no_hop_answers():
    blob = (t.get_task_statement() + " " + " ".join(t.get_success_criteria())).lower()
    for leak in ("stevenson", "vaea", "upolu"):
        assert leak not in blob, f"task statement leaks {leak!r}"
    # The keystone figure must not appear in the statement either.
    assert "472" not in t.get_task_statement()
    assert "1,549" not in t.get_task_statement()
