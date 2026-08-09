"""
Fixture-grounded proof that ``_compose_argmax``'s RENDER satisfies the real, already-shipped
validators of task 062 — offline, free, no LLM.

062 is scored ENTIRELY by grep validators (``get_llm_validation_function()`` returns None), so a
byte-level render mismatch is an unrecoverable 0 with no prose-quality cushion. The unit tests in
``execution_compiled_test.py`` pin the render string; this file closes the loop from the other side:
build synthetic per-leaf ``results`` shaped like real ``"<value> — source: <url>"`` leaf output
straight from 062's own ``ENTITIES`` fixture, run the composer, and feed the composed deliverable to
the module's REAL validator functions. Sibling of
``execution_compiled_and_filter_count_threshold_validators_test.py`` (072/076/078).

Why this task: the argmax is exactly the step a weak model gets wrong in free text — one stored
trace prints "Jengish Chokusu: 4,147 m" as the largest figure in its own list and then names a
SMALLER entry as the winner. Computing the comparison in Python removes that class of slip entirely
without touching how the six figures are gathered.

``observability`` carries a synthetic ``{"visit": {"count": N}}`` because 062's keystone is
grounding-gated (a peak name alone must never earn credit) and its coverage score is capped by the
visit count; the real run's visits happen in the leaves, upstream of and independent of composition.
"""
from agent.app.testing import execution_compiled as ec
from agent.app.idea_tests import test_062_tier5_prominence_argmax as t062


_WIKI = "https://en.wikipedia.org/wiki/"


def _results(overrides=None, unknown=()):
    """One fact per peak, exactly as ``_run_leaf_thin`` returns it (``UNKNOWN — <url>`` on a miss)."""
    overrides = overrides or {}
    out = {}
    for e in t062.ENTITIES:
        url = f"{_WIKI}{e['name'].replace(' ', '_')}"
        out[e["key"]] = (f"UNKNOWN — {url}" if e["key"] in unknown
                         else f"{overrides.get(e['key'], e['prominence']):,} m — source: {url}")
    return out


def _compose(results):
    plan = t062.get_compiled_plan()
    return ec._compose(plan["leaves"], results, plan["composition"])


def _score_all(composed, n_visits):
    """Every real validator's verdict for a composed deliverable, keyed by check name."""
    result = {"output": {"final_deliverable": composed}}
    observability = {"visit": {"count": n_visits}}
    return {v["check"]: v for v in
            (check(result, observability) for check in t062.get_validation_functions())}


def test_062_composed_answer_passes_every_real_validator():
    composed = _compose(_results())
    assert composed is not None
    scores = _score_all(composed, len(t062.ENTITIES))
    assert all(v["score"] == 1.0 for v in scores.values()), scores
    # The winner is the composed argmax, not a coincidence of the fixture text.
    assert composed.startswith(f"{t062.WINNER['name']} has the highest topographic prominence")


def test_062_composed_answer_satisfies_the_keystone_assertion_regex_directly():
    """The lead sentence is accepted by ``_JENGISH_WINS`` itself — the winner tied to a superlative
    inside the proximity window, with no rival peak name intervening — so a future tightening of the
    keystone gate cannot silently zero the task."""
    composed = _compose(_results())
    assert t062._JENGISH_WINS.search(t062._strip_urls(composed))


def test_062_misread_prominence_moves_the_winner_and_fails_the_keystone():
    """Kongur Tagh (the ELEVATION decoy, 3,585 m) misread as 5,585 m becomes the argmax. The
    keystone must score 0: composition removes the LLM's comparison slips, it does not launder a bad
    extraction into credit."""
    composed = _compose(_results({"kongur_tagh": 5585}))
    assert composed.startswith("Kongur Tagh has the highest topographic prominence")
    scores = _score_all(composed, len(t062.ENTITIES))
    assert scores["keystone_argmax"]["score"] == 0.0
    # ... and the gated secondaries short-circuit with it.
    assert scores["winner_prominence"]["score"] == 0.0
    assert scores["citation"]["score"] == 0.0


def test_062_one_unresolved_peak_still_scores_the_keystone():
    """DEGRADE GRACEFULLY: unlike the count/AND-filter tasks, a missing figure only costs its own
    coverage/citation row — the argmax over the remaining five is still the right answer, and the
    unresolved peak is disclosed as UNKNOWN rather than dropped."""
    composed = _compose(_results(unknown=("noshaq",)))
    assert composed.endswith("Noshaq: topographic prominence=UNKNOWN")
    scores = _score_all(composed, len(t062.ENTITIES))
    assert scores["keystone_argmax"]["score"] == 1.0
    assert scores["winner_prominence"]["score"] == 1.0
    assert scores["coverage"]["score"] == 5 / 6      # honest partial credit, not a fabricated 6/6


def test_062_declines_when_fewer_than_two_peaks_resolve():
    """A 'highest of the six' claim backed by ONE figure is vacuous -> None -> the unchanged
    free-text fallback runs. This is the composition-side mirror of the vote-quorum rule."""
    others = tuple(e["key"] for e in t062.ENTITIES if not e["winner"])
    assert _compose(_results(unknown=others)) is None
    assert _compose(_results(unknown=tuple(e["key"] for e in t062.ENTITIES))) is None


def test_062_plan_structure_is_unchanged_by_the_composition_keys():
    """``composition``/``agg_mode`` are siblings of ``leaves``/``aggregation``: the compiled DAG the
    benchmark measures (six independent leaves, no edges) is byte-identical to before."""
    from agent.app.testing.compiled_plan import plan_structure, validate_plan
    plan = t062.get_compiled_plan()
    structure = plan_structure(plan)
    assert structure["leaf_count"] == len(t062.ENTITIES) == 6
    assert structure["edge_count"] == 0
    norm = validate_plan(plan)
    assert [leaf["id"] for leaf in norm["leaves"]] == [e["key"] for e in t062.ENTITIES]
    # No prominence figure and no winner leak into the scaffold.
    scaffold = repr(plan["leaves"]) + plan["aggregation"] + repr(plan["composition"])
    assert not any(str(e["prominence"]) in scaffold for e in t062.ENTITIES)
