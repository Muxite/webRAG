"""
Fixture-grounded proof that ``_compose_ratio_argmax``'s RENDER satisfies the real, already-shipped
validators of task 064 — offline, free, no LLM.

064 is scored ENTIRELY by grep validators (``get_llm_validation_function()`` returns None), so a
byte-level render mismatch is an unrecoverable 0 with no prose-quality cushion. The unit tests in
``execution_compiled_test.py`` pin the render string against a self-contained fixture; this file
closes the loop from the other side: build synthetic per-leaf ``results`` shaped like real
thin-leaf output straight from 064's own ``ENTITIES`` fixture, run the composer, and feed the
composed deliverable to the module's REAL validator functions. Sibling of
``execution_compiled_argmax_validators_test.py`` (062's argmax precedent).

Why this task specifically: 064 is the first composer whose leaf gathers TWO labelled numbers off
ONE page in a single free-text fact string — the extraction load-bearing piece
(``_field_number``/``_extract_ratio_pair``) has no analogue in the plain single-number composers, so
this file proves the Tier-A-only (both fields must be explicitly labelled) extraction survives
contact with the REAL keystone gate, coverage diagnostic, and gated secondaries — not just
hand-picked unit fixtures — and documents the residual risks the design explicitly did NOT close
(mislabelled-but-present data).

``observability`` carries a synthetic ``{"visit": {"count": N}}`` because 064's keystone is
grounding-gated (a name alone must never earn credit) and its coverage/citation scores are capped by
the visit count; the real run's visits happen in the leaves, upstream of and independent of
composition.
"""
from agent.app.testing import execution_compiled as ec
from agent.app.idea_tests import test_064_tier5_computed_ratio_argmax_wide as t064


_WIKI = "https://en.wikipedia.org/wiki/"


def _slug(name: str) -> str:
    return name.replace(" ", "_")


def _fact(volume, area, name, swapped=False):
    """One leaf's fact, shaped like real thin-leaf output: both figures explicitly labelled."""
    url = f"{_WIKI}{_slug(name)}"
    if swapped:
        volume, area = area, volume
    return f"Volume: {volume:,} km^3, Area: {area:,} km^2 — source: {url}"


def _results(overrides=None, unknown=(), swapped=()):
    """One fact per lake, straight from ``t064.ENTITIES`` unless overridden/swapped/blanked."""
    overrides = overrides or {}
    out = {}
    for e in t064.ENTITIES:
        if e["key"] in unknown:
            out[e["key"]] = f"UNKNOWN — {_WIKI}{_slug(e['name'])}"
            continue
        volume, area = overrides.get(e["key"], (e["volume"], e["area"]))
        out[e["key"]] = _fact(volume, area, e["name"], swapped=e["key"] in swapped)
    return out


def _compose(results):
    plan = t064.get_compiled_plan()
    return ec._compose(plan["leaves"], results, plan["composition"])


def _score_all(composed, n_visits):
    """Every real validator's verdict for a composed deliverable, keyed by check name."""
    result = {"output": {"final_deliverable": composed}}
    observability = {"visit": {"count": n_visits}}
    return {v["check"]: v for v in
            (check(result, observability) for check in t064.get_validation_functions())}


def test_064_composed_answer_passes_every_real_validator():
    composed = _compose(_results())
    assert composed is not None
    scores = _score_all(composed, len(t064.ENTITIES))
    assert all(v["score"] == 1.0 for v in scores.values()), scores
    # The winner is the composed argmax, not a coincidence of the fixture text.
    assert composed.startswith(f"{t064.WINNER['name']} has the highest volume-to-surface-area ratio")


def test_064_composed_answer_satisfies_the_keystone_assertion_regex_directly():
    """The lead sentence is accepted by ``_ISSYK_WINS`` itself — the winner tied to a superlative
    inside the proximity window, with no rival lake name intervening — so a future tightening of the
    keystone gate cannot silently zero the task."""
    composed = _compose(_results())
    assert t064._ISSYK_WINS.search(t064._strip_urls(composed))


def test_064_swapped_fields_for_a_non_winning_lake_is_a_documented_residual_risk():
    """The composer trusts EXPLICIT labels — it cannot detect a model that mislabels its OWN
    correct numbers (only unlabelled-decoy-guessing is what adversarial review closed). Lake
    Victoria's volume/area are BOTH explicitly labelled but SWAPPED: this computes an absurdly
    inflated ratio and WINS the composition outright — a concrete demonstration of the residual risk
    the design doc flags (section 7's fixture-grounded test list), not a hypothetical. Mislabelled-
    but-present data is a different failure mode from the closed unlabelled-decoy-guessing one, and
    it is NOT separately guarded against."""
    composed = _compose(_results(swapped=("victoria",)))
    assert composed is not None
    swapped_ratio = round((59947 / 2424) * 1000, 1)
    assert composed.startswith(
        f"Lake Victoria has the highest volume-to-surface-area ratio of the 5 lakes compared, "
        f"at {ec._fmt_num(swapped_ratio)} m."
    )
    scores = _score_all(composed, len(t064.ENTITIES))
    # The residual risk materializes concretely: the keystone (Issyk-Kul) is no longer credited.
    assert scores["keystone_argmax"]["score"] == 0.0


def test_064_unlabelled_bare_numbers_for_the_winner_are_unresolved_not_guessed():
    """If Issyk-Kul's OWN fact drops both labels and just states two bare numbers, Tier-A-only
    refuses (no positional fallback) -- that lake becomes UNKNOWN rather than silently paired in
    the wrong order. ALL-OR-NOTHING (changed after a live regression, see execution_compiled_test's
    true-winner-missing test): the composition now refuses ENTIRELY rather than firing over the
    other four -- falling back to the unchanged free-text path is safer than a partial render that
    is missing the very lake the keystone check requires."""
    bare = dict(_results(), issyk_kul=f"1,736 and 6,236 — source: {_WIKI}Issyk-Kul")
    composed = _compose(bare)
    assert composed is None


def test_064_erie_glaciation_decoy_is_refused_not_papered_over():
    """The exact Erie/formation-year decoy from the adversarial review, fed through the FULL
    composer (not just ``_extract_ratio_pair`` in isolation): Tier-A-only correctly refuses to pair
    the decoy (proving the field-extraction safety holds), and then the ALL-OR-NOTHING threshold
    (changed after a live regression) means Erie's single unresolved row sinks the WHOLE
    composition rather than degrading to a 4/5 partial render -- falls back to free text instead of
    risking a partial answer that silently drops a declared item."""
    decoy_text = (f"Area: 25,667 km^2. The lake formed about 10,000 years ago after the last "
                  f"glaciation. — source: {_WIKI}Lake_Erie")
    results = dict(_results(), erie=decoy_text)
    composed = _compose(results)
    assert composed is None


def test_064_plan_structure_is_unchanged_by_the_composition_keys():
    """``composition``/``agg_mode`` are siblings of ``leaves``/``aggregation``: the compiled DAG the
    benchmark measures (five independent leaves, no edges) is byte-identical to before."""
    from agent.app.testing.compiled_plan import plan_structure, validate_plan
    plan = t064.get_compiled_plan()
    structure = plan_structure(plan)
    assert structure["leaf_count"] == len(t064.ENTITIES) == 5
    assert structure["edge_count"] == 0
    norm = validate_plan(plan)
    assert [leaf["id"] for leaf in norm["leaves"]] == [e["key"] for e in t064.ENTITIES]
    # No volume/area figure leaks into the scaffold beyond the task's own given lake names/regions.
    scaffold = repr(plan["leaves"]) + plan["aggregation"] + repr(plan["composition"])
    assert not any(str(e["volume"]) in scaffold or str(e["area"]) in scaffold
                   for e in t064.ENTITIES)
