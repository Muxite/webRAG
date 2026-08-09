"""
Offline unit tests for the Icelandic-tunnel count-with-condition task (test 090) — free.

Mirrors the adversarial suite shared with 072/073/087 for the COUNT keystone
(KEYSTONE_COUNT = 4 tunnels with length > 4,500 m, reported in deliverables[0]):

  * a correct full answer scores 1.0 on every validator (single- AND multi-line layout);
  * a wrong count (3, 5, or the naive 'all six' 6) gates the keystone to 0 AND short-circuits the
    gated secondaries (passing-tunnel names, citation) to 0, while the UN-gated coverage diagnostic
    is retained (all six lengths still credited);
  * partial coverage yields the exact gathered fraction;
  * a no-visits run gates the visit process metric to 0;
  * the compiled plan is a well-formed 6-way independent fan-out (6 leaves, 0 edges) whose leaves
    self-describe their entity and which leaks no length value.
"""
from agent.app.idea_tests import test_090_tier5_tunnel_count_iceland as t
from agent.app.testing import compiled_plan as cp


_OBS = {"visit": {"count": 6}}

# Body carrying all six lengths, all four passing names, and all six source URLs.
_BODY_SINGLE = (
    "Passing (> 4,500 m): Vaðlaheiðargöng 7,400 m, Hvalfjörður Tunnel 5,770 m, "
    "Dýrafjarðargöng 5,600 m, Bolungarvíkurgöng 5,426 m. "
    "Failing: Múlagöng 3,400 m, Almannaskarðsgöng 1,300 m. "
    "Sources: https://en.wikipedia.org/wiki/Va%C3%B0lahei%C3%B0arg%C3%B6ng "
    "https://en.wikipedia.org/wiki/Hvalfj%C3%B6r%C3%B0ur_Tunnel "
    "https://en.wikipedia.org/wiki/D%C3%BDrafjar%C3%B0arg%C3%B6ng "
    "https://en.wikipedia.org/wiki/Bolungarv%C3%ADkurg%C3%B6ng "
    "https://en.wikipedia.org/wiki/M%C3%BAlag%C3%B6ng "
    "https://en.wikipedia.org/wiki/Almannaskar%C3%B0sg%C3%B6ng"
)

_BODY_MULTI = (
    "Restated lengths:\n"
    "  Vaðlaheiðargöng: length = 7,400 m — EXCEEDS 4,500 m\n"
    "  Hvalfjörður Tunnel: length = 5,770 m — EXCEEDS 4,500 m\n"
    "  Dýrafjarðargöng: length = 5,600 m — EXCEEDS 4,500 m\n"
    "  Bolungarvíkurgöng: length = 5,426 m — EXCEEDS 4,500 m\n"
    "  Múlagöng: length = 3,400 m — below 4,500 m\n"
    "  Almannaskarðsgöng: length = 1,300 m — below 4,500 m\n"
    "Sources:\n"
    "  https://en.wikipedia.org/wiki/Va%C3%B0lahei%C3%B0arg%C3%B6ng\n"
    "  https://en.wikipedia.org/wiki/Hvalfj%C3%B6r%C3%B0ur_Tunnel\n"
    "  https://en.wikipedia.org/wiki/D%C3%BDrafjar%C3%B0arg%C3%B6ng\n"
    "  https://en.wikipedia.org/wiki/Bolungarv%C3%ADkurg%C3%B6ng\n"
    "  https://en.wikipedia.org/wiki/M%C3%BAlag%C3%B6ng\n"
    "  https://en.wikipedia.org/wiki/Almannaskar%C3%B0sg%C3%B6ng\n"
)


def _result(count_primary: str, body: str):
    """Build a result whose deliverables[0] is the keystone (count) slot and deliverables[1] is the
    supporting body (names, lengths, URLs)."""
    return {
        "deliverables": [count_primary, body],
        "output": {"final_deliverable": f"{count_primary}. {body}"},
    }


# ── correct full answer ────────────────────────────────────────────────────────

def test_full_answer_single_line_scores_all():
    r = _result("4", _BODY_SINGLE)
    assert t.validate_keystone_count(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_passing_tunnels(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    # Same answer in a newline-heavy layout: keystone, coverage, names, citations all register.
    r = _result("4", _BODY_MULTI)
    assert t.validate_keystone_count(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_passing_tunnels(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0


# ── wrong keystone: gate to 0, coverage retained ───────────────────────────────

def test_wrong_count_five_gates_keystone_and_secondaries_but_keeps_coverage():
    # Miscounting Múlagöng (3,400 m) as passing -> count 5 (wrong).
    r = _result("5", _BODY_MULTI)
    assert t.validate_keystone_count(r, _OBS)["passed"] is False
    assert t.validate_keystone_count(r, _OBS)["score"] == 0.0
    # Gated secondaries short-circuit to 0 despite the body naming the correct four + their URLs.
    assert t.validate_passing_tunnels(r, _OBS)["score"] == 0.0
    assert t.validate_citation(r, _OBS)["score"] == 0.0
    # UN-gated coverage is retained: all six lengths were still gathered.
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


def test_dropped_entity_count_three_fails_keystone():
    r = _result("3", _BODY_MULTI)
    assert t.validate_keystone_count(r, _OBS)["passed"] is False


def test_naive_all_six_fails_keystone():
    r = _result("6", _BODY_MULTI)
    assert t.validate_keystone_count(r, _OBS)["passed"] is False


def test_threshold_token_does_not_masquerade_as_count():
    # A wrong-count answer that also states the threshold (4,500 m / 4.5 km) must still fail:
    # 4,500 -> 4500 and 4.5 -> dropped decimal, so no bare 4 leaks into the keystone check.
    r = _result("3 tunnels exceed 4,500 m (4.5 km)", _BODY_MULTI)
    assert t.validate_keystone_count(r, _OBS)["passed"] is False


# ── grounding gate ──────────────────────────────────────────────────────────────

def test_ungrounded_correct_count_gates_to_zero():
    """Grounding requirement: the correct keystone COUNT alone must NOT earn credit if the agent
    never actually visited a page (visit.count == 0) — an ungrounded parametric-memory guess must
    collapse the keystone gate (and everything gated on it) to 0, not just the value match."""
    r = _result("4", _BODY_SINGLE)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_count(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_count(r, ungrounded_obs)["passed"] is False
    assert t.validate_passing_tunnels(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citation(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_count(r, ungrounded_obs)["score"],
        t.validate_coverage(r, ungrounded_obs)["score"],
        t.validate_passing_tunnels(r, ungrounded_obs)["score"],
        t.validate_citation(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


# ── partial coverage yields exact fraction ─────────────────────────────────────

def test_partial_coverage_scores_fraction():
    # Correct count keystone, but only four of the six (tunnel, length) pairs reported.
    partial = (
        "Vaðlaheiðargöng 7,400 m, Hvalfjörður Tunnel 5,770 m, "
        "Dýrafjarðargöng 5,600 m, Múlagöng 3,400 m."
    )
    r = _result("4", partial)
    assert t.validate_keystone_count(r, _OBS)["score"] == 1.0
    cov = t.validate_coverage(r, _OBS)
    assert abs(cov["score"] - 4.0 / 6.0) < 1e-9
    assert cov["passed"] is False


# ── process metric ─────────────────────────────────────────────────────────────

def test_no_visits_gates_visit_metric():
    r = _result("4", _BODY_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_visits(r, {"visit": {"count": 0}})["passed"] is False
    # 3 of 6 visits -> half credit, still below the >=5 pass bar.
    assert abs(t.validate_visits(r, {"visit": {"count": 3}})["score"] - 0.5) < 1e-9
    assert t.validate_visits(r, {"visit": {"count": 3}})["passed"] is False


# ── compiled plan ──────────────────────────────────────────────────────────────

def test_compiled_plan_is_well_formed_six_way_fanout():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)  # must not raise (well-formed, acyclic, deps resolve)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 6
    assert struct["edge_count"] == 0  # six independent parallel leaves, no cross-entity hops


def test_compiled_plan_leaves_self_describe_their_entity():
    plan = t.get_compiled_plan()
    # Each leaf's own answer spec ('expect') must restate its tunnel name so the aggregator can
    # bind the length back to the right entity even after leaf IDs are stripped.
    for leaf, e in zip(plan["leaves"], t.ENTITIES):
        assert e["name"] in leaf["expect"], f"leaf {leaf['id']} does not self-describe its entity"


def test_compiled_plan_leaks_no_length_value():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    # STRUCTURE only: the plan names the six GIVEN tunnels and the GIVEN threshold (4,500 m / 4.5 km),
    # but leaks no length value and no per-entity membership.
    for tok in (
        "7,400", "7400", "7.4",
        "5,770", "5770", "5.77",
        "5,600", "5600", "5.6",
        "5,426", "5426", "5.426",
        "3,400", "3400", "3.4",
        "1,300", "1300", "1.3",
    ):
        assert tok not in blob, f"plan leaks length token {tok!r}"
