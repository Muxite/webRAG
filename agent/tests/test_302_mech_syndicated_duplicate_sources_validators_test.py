"""
Offline adversarial unit tests for the syndicated-duplicate-URL mechanism task (test 302) —
free, no LLM, no network.

The defect under test: treating three domains that all republish ONE English-Wikipedia
article as three independent corroborating sources. These tests pin that

  * a correct answer scores 1.0 in BOTH a single-line and a multi-line report layout,
  * the naive "three independent sources confirm it" answer gates the keystone (and the two
    keystone-gated secondaries) to exactly 0 while KEEPING its un-gated coverage score,
  * partial coverage scores the exact fraction,
  * a correct-but-ungrounded (no-visit) answer earns no keystone credit,
  * the compiled plan is well-formed and leaks no part of the answer.
"""
from agent.app.idea_tests import test_302_mech_syndicated_duplicate_sources as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 4}}

# --- a correct answer, multi-line report layout -----------------------------------------
_CORRECT_MULTILINE = (
    "Independent sources among the three given URLs: 1\n"
    "https://en.wikipedia.org/wiki/Crater_Lake prints 1,949 feet (594 m) - this is the origin.\n"
    "https://alchetron.com/Crater-Lake prints 1,949 feet (594 m); its footer credits "
    "'Crater Lake Wikipedia' CC BY-SA, so it is a republished copy of the Wikipedia article.\n"
    "https://dbpedia.org/page/Crater_Lake gives dbo:maximumDepth 594.055200 and states the "
    "content was extracted from Wikipedia, so it is a mirror of the same article.\n"
    "Verdict: these three URLs are NOT three independent sources - they are one source "
    "republished on three domains.\n"
    "Independent check: the National Park Service page "
    "https://www.nps.gov/crla/learn/nature/crater-lake.htm prints 1,943 ft (592 m), which "
    "does not match the claim of 1,949 ft.\n"
    "Proof of visit (nps.gov): Deep Rover made 47 dives to the bottom of the lake.\n"
)

# --- the same content, single-line/prose layout ------------------------------------------
_CORRECT_SINGLELINE = (
    "The three given URLs (en.wikipedia.org/wiki/Crater_Lake, alchetron.com/Crater-Lake and "
    "dbpedia.org/page/Crater_Lake) all print the same 1,949 feet / 594 m figure, but "
    "alchetron and dbpedia both credit Wikipedia as the origin of their text, so they are "
    "copies of one article and amount to only one independent source, not three. "
    "The genuinely independent authority, the National Park Service at "
    "https://www.nps.gov/crla/learn/nature/crater-lake.htm, prints 1,943 ft instead, so the "
    "claim is not corroborated. Deep Rover made 47 dives to the bottom, per that same page."
)

# --- the defect: three copies read as three independent confirmations --------------------
_NAIVE_FALSE_CORROBORATION = (
    "Maximum depth: 1,949 feet (594 m).\n"
    "This is confirmed by three independent sources:\n"
    "1. https://en.wikipedia.org/wiki/Crater_Lake - 1,949 feet (594 m)\n"
    "2. https://alchetron.com/Crater-Lake - 1,949 feet (594 m)\n"
    "3. https://dbpedia.org/page/Crater_Lake - 594.055200 m\n"
    "The National Park Service page https://www.nps.gov/crla/learn/nature/crater-lake.htm "
    "says 1,943 ft, but three sources agree on 1,949 so the claim stands. "
    "Deep Rover made 47 dives to the bottom.\n"
)


def test_correct_answer_scores_full_multiline():
    r = _r(_CORRECT_MULTILINE)
    assert t.validate_keystone_source_independence(r, _OBS)["score"] == 1.0
    assert t.validate_nps_independent_figure(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0
    scores = [f(r, _OBS)["score"] for f in t.get_validation_functions()]
    assert sum(scores) / len(scores) == 1.0


def test_correct_answer_scores_full_single_line_layout():
    r = _r(_CORRECT_SINGLELINE)
    assert t.validate_keystone_source_independence(r, _OBS)["score"] == 1.0
    assert t.validate_nps_independent_figure(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    scores = [f(r, _OBS)["score"] for f in t.get_validation_functions()]
    assert sum(scores) / len(scores) >= 0.75


def test_false_corroboration_gates_keystone_and_secondaries_but_keeps_coverage():
    """THE mechanism assertion: three syndicated copies claimed as independent -> keystone 0."""
    r = _r(_NAIVE_FALSE_CORROBORATION)
    assert t.validate_keystone_source_independence(r, _OBS)["score"] == 0.0
    assert t.validate_keystone_source_independence(r, _OBS)["passed"] is False
    # Secondaries short-circuit even though 1,943 and the URLs are all present.
    assert t.validate_nps_independent_figure(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0
    # Un-gated breadth diagnostic is retained in full: the agent DID gather everything.
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    scores = [f(r, _OBS)["score"] for f in t.get_validation_functions()]
    assert sum(scores) / len(scores) < 0.75


def test_silent_omission_also_gates_keystone():
    """No corroboration claim at all, but no independence verdict either -> keystone 0."""
    text = (
        "Crater Lake maximum depth is 1,949 feet (594 m) per "
        "https://en.wikipedia.org/wiki/Crater_Lake, https://alchetron.com/Crater-Lake and "
        "https://dbpedia.org/page/Crater_Lake. The NPS page "
        "https://www.nps.gov/crla/learn/nature/crater-lake.htm gives 1,943 ft. "
        "Deep Rover made 47 dives."
    )
    r = _r(text)
    assert t.validate_keystone_source_independence(r, _OBS)["score"] == 0.0
    assert t.validate_nps_independent_figure(r, _OBS)["score"] == 0.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


def test_neutral_factual_statement_is_not_read_as_the_defect():
    """'All three pages print 1,949 ft' is an observation, not a corroboration claim."""
    text = (
        "All three pages print 1,949 feet (594 m). However alchetron and dbpedia both "
        "attribute their text to the Wikipedia article, so this is one source republished, "
        "i.e. only one independent source. nps.gov states 1,943 ft. Deep Rover: 47 dives."
    )
    r = _r(text)
    assert t.validate_keystone_source_independence(r, _OBS)["score"] == 1.0


def test_partial_coverage_scores_exact_fraction():
    """Only the Wikipedia leg gathered: 3 of the 6 coverage items (figure, wikipedia, ...)."""
    text = (
        "https://en.wikipedia.org/wiki/Crater_Lake gives 1,949 feet (594 m). "
        "https://alchetron.com/Crater-Lake shows the same figure and credits Wikipedia, so "
        "it is a copy - one independent source so far. I did not reach the other pages."
    )
    r = _r(text)
    cov = t.validate_coverage(r, _OBS)
    assert cov["score"] == 3 / 6  # claimed figure + wikipedia + alchetron
    assert cov["passed"] is False
    # Keystone still stands (duplication recognised); the missing independent check does not.
    assert t.validate_keystone_source_independence(r, _OBS)["score"] == 1.0
    assert t.validate_nps_independent_figure(r, _OBS)["score"] == 0.0


def test_visit_gate_and_ungrounded_answer():
    r = _r(_CORRECT_MULTILINE)
    no_visits = {"visit": {"count": 0}}
    assert t.validate_visits(r, no_visits)["score"] == 0.0
    assert t.validate_visits(r, no_visits)["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 4}})["passed"] is True
    # Ungrounded: keystone + gated secondaries collapse, un-gated coverage is retained.
    assert t.validate_keystone_source_independence(r, no_visits)["score"] == 0.0
    assert t.validate_nps_independent_figure(r, no_visits)["score"] == 0.0
    assert t.validate_citations(r, no_visits)["score"] == 0.0
    assert t.validate_coverage(r, no_visits)["score"] == 1.0
    scores = [f(r, no_visits)["score"] for f in t.get_validation_functions()]
    assert sum(scores) / len(scores) < 0.75


def test_nps_figure_must_be_attributed_not_merely_present():
    """1,943 appears in a Wikipedia reference entry, so a bare mention earns nothing."""
    text = _CORRECT_MULTILINE.replace(
        "Independent check: the National Park Service page "
        "https://www.nps.gov/crla/learn/nature/crater-lake.htm prints 1,943 ft (592 m), which "
        "does not match the claim of 1,949 ft.\n",
        "One of the Wikipedia references lists 1,943 feet.\n",
    )
    r = _r(text)
    assert t.validate_keystone_source_independence(r, _OBS)["score"] == 1.0
    assert t.validate_nps_independent_figure(r, _OBS)["score"] == 0.0


def test_metadata_and_statement_contract():
    md = t.get_test_metadata()
    assert md["test_id"] == "302"
    assert md["level"] in {"micro", "integration", "navigation", "graph"}
    assert t.get_llm_validation_function() is None
    stmt = t.get_task_statement()
    for url in (e["url"] for e in t.SYNDICATED_URLS):
        assert url in stmt
    # The mandate must trigger the grounding gate (explicit visit + do-not-guess language).
    assert "do not guess" in stmt.lower() and "open" in stmt.lower()
    assert len(t.get_required_deliverables()) >= 4
    assert len(t.get_success_criteria()) >= 4


def test_compiled_plan_validates_and_leaks_nothing():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    # No answer leakage: not the verdict, not either depth figure, not the dive count.
    # (The en.wikipedia.org URL itself is a GIVEN from the mandate, so it is not a leak; the
    # plan must never say those pages are *copies* of it.)
    for leak in ("1,943", "1943", "1,949", "1949", "594", "592", "mirror", "syndicat",
                 " 47 ", "duplicat", "copy", "copies", "republish", "not independent"):
        assert leak not in blob, f"compiled plan leaks {leak!r}"
    ids = [l["id"] for l in plan["leaves"]]
    assert len(ids) == len(set(ids))
    # The dependent leaves chain off the discovered agency URL via {dep_id} templating.
    dependent = [l for l in plan["leaves"] if l.get("depends_on")]
    assert len(dependent) == 2
    for leaf in dependent:
        assert leaf["depends_on"] == ["agency_page_url"]
        assert "{agency_page_url}" in leaf["instruction"]
