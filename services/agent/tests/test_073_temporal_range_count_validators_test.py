"""
Offline unit tests for the temporal range-filter COUNT task (test 073) — free.

Covers the reworked COUNT keystone (IN_RANGE_COUNT = 3 in deliverables[0]) that replaced the old
"name the exact in-range subset" keystone (too hard for a cheap compiled-leaf executor per the
count-with-condition design law shared with tests 072/078/082/087/089):

  * a correct full answer scores 1.0 on every validator (single- AND multi-line layout);
  * a hallucinated / wrong count (2 or 4) gates the keystone to 0 AND short-circuits the gated
    secondaries (in-range names, in-range years, citation) to 0, while the UN-gated coverage
    diagnostic is retained (all six founding years still credited);
  * a naive "all six / 6" and a dropped-entity "2" both fail the keystone;
  * partial coverage yields the exact gathered fraction;
  * a no-visits run gates the visit process metric to 0;
  * the compiled plan is a well-formed 6-way independent fan-out (6 leaves, 0 edges) that leaks no
    founding year.
"""
from agent.app.idea_tests import test_073_tier5_temporal_range_filter as t
from agent.app.testing import compiled_plan as cp


_OBS = {"visit": {"count": 6}}

# Body carrying all six founding years, all three in-range names, and all six source URLs.
_BODY_SINGLE = (
    "In-range institutions: National Theatre of Kosovo (founded 1946), "
    "Kosovo Museum (established 1949), Luigj Gurakuqi University of Shkodra (established 1957). "
    "Out of range: Skopje Zoo 1926, Academy of Sciences of Albania 1972, "
    "National History Museum (Albania) 1981. "
    "Sources: https://en.wikipedia.org/wiki/National_Theatre_of_Kosovo "
    "https://en.wikipedia.org/wiki/Kosovo_Museum "
    "https://en.wikipedia.org/wiki/Luigj_Gurakuqi_University_of_Shkod%C3%ABr "
    "https://en.wikipedia.org/wiki/Skopje_Zoo "
    "https://en.wikipedia.org/wiki/Academy_of_Sciences_of_Albania "
    "https://en.wikipedia.org/wiki/National_History_Museum_(Albania)"
)

_BODY_MULTI = (
    "Restated founding years:\n"
    "  National Theatre of Kosovo: founded 1946 — IN RANGE\n"
    "  Kosovo Museum: founded 1949 — IN RANGE\n"
    "  Luigj Gurakuqi University of Shkodra: founded 1957 — IN RANGE\n"
    "  Skopje Zoo: founded 1926 — OUT OF RANGE\n"
    "  Academy of Sciences of Albania: founded 1972 — OUT OF RANGE\n"
    "  National History Museum (Albania): founded 1981 — OUT OF RANGE\n"
    "Sources:\n"
    "  https://en.wikipedia.org/wiki/National_Theatre_of_Kosovo\n"
    "  https://en.wikipedia.org/wiki/Kosovo_Museum\n"
    "  https://en.wikipedia.org/wiki/Luigj_Gurakuqi_University_of_Shkod%C3%ABr\n"
    "  https://en.wikipedia.org/wiki/Skopje_Zoo\n"
    "  https://en.wikipedia.org/wiki/Academy_of_Sciences_of_Albania\n"
    "  https://en.wikipedia.org/wiki/National_History_Museum_(Albania)\n"
)


def _result(count_primary: str, body: str):
    """Build a result whose deliverables[0] is the keystone (count) slot and deliverables[1] is the
    supporting body (names, years, URLs)."""
    return {
        "deliverables": [count_primary, body],
        "output": {"final_deliverable": f"{count_primary}. {body}"},
    }


# ── correct full answer ────────────────────────────────────────────────────────

def test_full_answer_single_line_scores_all():
    r = _result("3", _BODY_SINGLE)
    assert t.validate_keystone_count(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_inrange_names(r, _OBS)["score"] == 1.0
    assert t.validate_inrange_years(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    # Same answer in a newline-heavy layout: keystone, coverage, names, years, citations all register.
    r = _result("3", _BODY_MULTI)
    assert t.validate_keystone_count(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_inrange_names(r, _OBS)["score"] == 1.0
    assert t.validate_inrange_years(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0


# ── wrong keystone: gate to 0, coverage retained ───────────────────────────────

def test_wrong_count_four_gates_keystone_and_secondaries_but_keeps_coverage():
    # Hallucinated year drags an out-of-range entity into the range -> count 4 (wrong).
    r = _result("4", _BODY_MULTI)
    assert t.validate_keystone_count(r, _OBS)["passed"] is False
    assert t.validate_keystone_count(r, _OBS)["score"] == 0.0
    # Gated secondaries short-circuit to 0 despite the body naming the correct three + their years.
    assert t.validate_inrange_names(r, _OBS)["score"] == 0.0
    assert t.validate_inrange_years(r, _OBS)["score"] == 0.0
    assert t.validate_citation(r, _OBS)["score"] == 0.0
    # UN-gated coverage is retained: all six founding years were still gathered.
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


def test_dropped_entity_count_two_fails_keystone():
    r = _result("2", _BODY_MULTI)
    assert t.validate_keystone_count(r, _OBS)["passed"] is False


def test_naive_all_six_fails_keystone():
    r = _result("6", _BODY_MULTI)
    assert t.validate_keystone_count(r, _OBS)["passed"] is False


# ── partial coverage yields exact fraction ─────────────────────────────────────

def test_partial_coverage_scores_fraction():
    # Correct count keystone, but only four of the six founding years reported.
    partial = (
        "National Theatre of Kosovo 1946, Kosovo Museum 1949, "
        "Luigj Gurakuqi University of Shkodra 1957, Skopje Zoo 1926."
    )
    r = _result("3", partial)
    assert t.validate_keystone_count(r, _OBS)["score"] == 1.0
    cov = t.validate_coverage(r, _OBS)
    assert abs(cov["score"] - 4.0 / 6.0) < 1e-9
    assert cov["passed"] is False


# ── process metric ─────────────────────────────────────────────────────────────

def test_no_visits_gates_visit_metric():
    r = _result("3", _BODY_SINGLE)
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


def test_compiled_plan_leaks_no_founding_year():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    # STRUCTURE only: the plan names the six GIVEN institutions and the GIVEN range bounds, but
    # leaks no founding year and no per-entity membership.
    for year in ("1946", "1949", "1957", "1926", "1972", "1981"):
        assert year not in blob, f"plan leaks founding year {year!r}"
