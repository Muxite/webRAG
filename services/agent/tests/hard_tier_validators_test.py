"""
Offline unit tests for the URL-free hard-tier task validators (tests 050 & 051) — free.

Confirm the keystone short-circuit and visit gate behave as designed on synthetic
deliverables, including that a no-visit answer loses the visit credit.
"""
from agent.app.idea_tests import test_050_tier3_search_chain as t3
from agent.app.idea_tests import test_051_tier4_dependent_chain as t4


def _r(text):
    return {"output": {"final_deliverable": text}}


# ---- Tier 3 (050) -------------------------------------------------------------

_T3_FULL = (
    "The novel 'Beloved' was written by Toni Morrison. She earned her master's (MA) degree "
    "from Cornell University. Sources: https://en.wikipedia.org/wiki/Toni_Morrison"
)


def test_t3_full_answer_scores_all():
    obs = {"visit": {"count": 2}}
    checks = {c["check"]: c for c in (
        t3.validate_visits(_r(_T3_FULL), obs),
        t3.validate_keystone_university(_r(_T3_FULL), obs),
        t3.validate_author(_r(_T3_FULL), obs),
        t3.validate_citation(_r(_T3_FULL), obs),
    )}
    assert checks["keystone_university"]["score"] == 1.0
    assert checks["author"]["score"] == 1.0
    assert checks["citation"]["score"] == 1.0
    assert checks["visit_count"]["score"] == 1.0


def test_t3_missing_keystone_short_circuits():
    obs = {"visit": {"count": 2}}
    r = _r("Beloved was written by Toni Morrison.")  # no university
    assert not t3.validate_keystone_university(r, obs)["passed"]
    assert t3.validate_author(r, obs)["score"] == 0.0
    assert t3.validate_citation(r, obs)["score"] == 0.0


# ---- Tier 4 (051) -------------------------------------------------------------

_T4_FULL = (
    "Things Fall Apart was written by Chinua Achebe, who attended University College Ibadan "
    "(now the University of Ibadan), which was founded in 1948. "
    "Sources: https://en.wikipedia.org/wiki/Chinua_Achebe and "
    "https://en.wikipedia.org/wiki/University_of_Ibadan"
)


def test_t4_full_answer_scores_all():
    obs = {"visit": {"count": 3}}
    checks = {c["check"]: c for c in (
        t4.validate_visits(_r(_T4_FULL), obs),
        t4.validate_keystone_year(_r(_T4_FULL), obs),
        t4.validate_chain_intermediate(_r(_T4_FULL), obs),
        t4.validate_chain_urls(_r(_T4_FULL), obs),
    )}
    assert checks["keystone_year"]["score"] == 1.0
    assert checks["chain_intermediate"]["score"] == 1.0
    assert checks["chain_urls"]["score"] == 1.0
    assert checks["visit_count"]["score"] == 1.0


def test_t4_no_visits_loses_visit_credit_even_if_year_present():
    obs = {"visit": {"count": 0}}
    assert t4.validate_keystone_year(_r(_T4_FULL), obs)["passed"]  # parametric leak possible
    assert t4.validate_visits(_r(_T4_FULL), obs)["score"] == 0.0  # but no evidence visits


def test_t4_missing_year_short_circuits():
    obs = {"visit": {"count": 3}}
    r = _r("Chinua Achebe attended the University of Ibadan.")  # no founding year
    assert not t4.validate_keystone_year(r, obs)["passed"]
    assert t4.validate_chain_intermediate(r, obs)["score"] == 0.0
    assert t4.validate_chain_urls(r, obs)["score"] == 0.0
