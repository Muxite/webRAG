"""
Fixture-grounded proof that ``_compose_argmax``'s RENDER satisfies the real, already-shipped
validators of task 077 -- offline, free, no LLM. Sibling of
``execution_compiled_argmax_validators_test.py`` (062), but with one extra regression class 062
doesn't need: 077's `value_label` was chosen as "main span" (not the more natural "longest main
span") specifically to avoid colliding with `_SUP`, this file's own superlative-trigger vocabulary
-- a live cross-model check found "longest main span" let the composer's own breakdown row
("Russky Bridge: longest main span=<value>") trivially satisfy the keystone regex regardless of
which bridge the lead sentence actually names, even with ZERO missing leaves. That collision is
the primary thing under test here, not just the composer mechanics already covered by 062's tests.

Motivation for wiring 077 at all: llama3.2:3b (live) correctly extracted all six real span figures
(Russky 1104m plainly present in its own list) and still concluded a SMALLER entry (Rio-Antirrio,
560m) was "the longest" -- the exact compute-right/conclude-wrong failure this mechanism exists to
eliminate, reproduced independently of the word-collision bug above.
"""
from agent.app.testing import execution_compiled as ec
from agent.app.idea_tests import test_077_tier5_pageonly_argmax as t077


_WIKI = "https://en.wikipedia.org/wiki/"


def _results(overrides=None, unknown=()):
    """One fact per bridge, exactly as ``_run_leaf_thin`` returns it (``UNKNOWN — <url>`` on a miss)."""
    overrides = overrides or {}
    out = {}
    for e in t077.ENTITIES:
        url = f"{_WIKI}{e['name'].replace(' ', '_')}"
        out[e["key"]] = (f"UNKNOWN — {url}" if e["key"] in unknown
                         else f"{overrides.get(e['key'], e['span']):,} m — source: {url}")
    return out


def _compose(results):
    plan = t077.get_compiled_plan()
    return ec._compose(plan["leaves"], results, plan["composition"])


def _score_all(composed, n_visits):
    """Every real validator's verdict for a composed deliverable, keyed by check name."""
    result = {"output": {"final_deliverable": composed}}
    observability = {"visit": {"count": n_visits}}
    return {v["check"]: v for v in
            (check(result, observability) for check in t077.get_validation_functions())}


def test_077_composed_answer_passes_every_real_validator():
    composed = _compose(_results())
    assert composed is not None
    scores = _score_all(composed, len(t077.ENTITIES))
    assert all(v["score"] == 1.0 for v in scores.values()), scores
    assert composed.startswith(f"{t077.WINNER['name']} has the highest main span")


def test_077_composed_answer_satisfies_the_keystone_assertion_regex_directly():
    composed = _compose(_results())
    assert t077._RUSSKY_WINS.search(composed)


def test_077_value_label_does_not_collide_with_sup_vocabulary():
    """The specific defect a naive wiring would reproduce: ``value_label`` must share no word with
    ``_SUP`` (this file's superlative-trigger list), or the composer's per-row breakdown
    ("<Name>: <value_label>=<value>") trivially satisfies the keystone regex for EVERY row,
    independent of which bridge actually won."""
    value_label = t077.get_compiled_plan()["composition"]["value_label"]
    sup_words = {w for w in t077._SUP.split("|")}
    label_words = set(value_label.lower().split())
    assert not (label_words & sup_words), (
        f"value_label {value_label!r} shares a word with _SUP: {label_words & sup_words}"
    )


def test_077_degraded_leaf_does_not_let_a_runner_up_steal_the_keystone():
    """THE regression this wiring was designed to avoid, reproduced directly: Russky Bridge's own
    leaf fails to resolve, but Edong/Tatara/the famous decoy Pont de Normandie all resolve cleanly.
    A word-collision bug (now fixed via ``value_label`` choice, see the test above) would have let
    the render's per-row breakdown wrongly satisfy the keystone even though the LEAD sentence
    names Edong (926m) as the winner, not Russky."""
    composed = _compose(_results(unknown=("russky", "rion_antirion", "helgeland")))
    assert composed.startswith("Edong Yangtze River Bridge has the highest main span")
    scores = _score_all(composed, len(t077.ENTITIES))
    assert scores["keystone_argmax"]["score"] == 0.0
    assert scores["winner_span"]["score"] == 0.0
    assert scores["citation"]["score"] == 0.0


def test_077_misread_span_moves_the_winner_and_fails_the_keystone():
    """Pont de Normandie (the FAME decoy, 856 m) misread as 9,856 m becomes the argmax. The
    keystone must score 0: composition removes the LLM's comparison slips, it does not launder a
    bad extraction into credit."""
    composed = _compose(_results({"normandie": 9856}))
    assert composed.startswith("Pont de Normandie has the highest main span")
    scores = _score_all(composed, len(t077.ENTITIES))
    assert scores["keystone_argmax"]["score"] == 0.0
    assert scores["winner_span"]["score"] == 0.0
    assert scores["citation"]["score"] == 0.0


def test_077_one_unresolved_bridge_still_scores_the_keystone():
    """DEGRADE GRACEFULLY (like 062): a single missing figure only costs its own coverage/citation
    row -- the argmax over the remaining five is still the right answer."""
    composed = _compose(_results(unknown=("helgeland",)))
    assert composed.endswith("Helgeland Bridge: main span=UNKNOWN")
    scores = _score_all(composed, len(t077.ENTITIES))
    assert scores["keystone_argmax"]["score"] == 1.0
    assert scores["winner_span"]["score"] == 1.0
    assert scores["coverage"]["score"] == 5 / 6


def test_077_declines_when_fewer_than_two_bridges_resolve():
    """A 'highest of the six' claim backed by ONE figure is vacuous -> None -> the unchanged
    free-text fallback runs."""
    others = tuple(e["key"] for e in t077.ENTITIES if not e["winner"])
    assert _compose(_results(unknown=others)) is None
    assert _compose(_results(unknown=tuple(e["key"] for e in t077.ENTITIES))) is None


def test_077_plan_structure_is_unchanged_by_the_composition_keys():
    """``composition``/``agg_mode`` are siblings of ``leaves``/``aggregation``: the compiled DAG the
    benchmark measures (six independent leaves, no edges) is byte-identical to before."""
    from agent.app.testing.compiled_plan import plan_structure, validate_plan
    plan = t077.get_compiled_plan()
    structure = plan_structure(plan)
    assert structure["leaf_count"] == len(t077.ENTITIES) == 6
    assert structure["edge_count"] == 0
    norm = validate_plan(plan)
    assert [leaf["id"] for leaf in norm["leaves"]] == [e["key"] for e in t077.ENTITIES]
    # No span FIGURE and no WINNER flag leak into the scaffold -- every bridge is named as a
    # label (legitimate: the task statement itself lists all six names), but which one wins and
    # what its value is must come from the leaves at runtime, not be pre-supplied here.
    scaffold = repr(plan["leaves"]) + plan["aggregation"] + repr(plan["composition"])
    assert not any(str(e["span"]) in scaffold for e in t077.ENTITIES)
    assert "winner" not in scaffold.lower()
