"""
Fixture-grounded proof that the deterministic composers' RENDER satisfies the real, already-shipped
validators of the three tasks that opt into ``agg_mode="computed"`` — offline, free, no LLM.

These three tasks are scored ENTIRELY by grep validators (``get_llm_validation_function()`` returns
None for all of them), so a byte-level render mismatch is an unrecoverable 0 with no prose-quality
cushion. The unit tests in ``execution_compiled_test.py`` pin the render string; this file closes
the loop from the other side: build synthetic per-leaf ``results`` shaped like real
``"<value> — source: <url>"`` leaf output straight from each module's own ``ENTITIES`` fixture, run
the composer, and feed the composed deliverable to the module's REAL validator functions.

  * 076 (and_filter, 12 leaves / 6 lakes / 2 constraints) -> every validator 1.0;
  * 072 (count_threshold, 7 lake depths)                  -> every validator 1.0;
  * 078 (count_threshold, 7 island areas, decimals)       -> every validator 1.0, incl. the
    UNROUNDED infobox figures (11,831.02 / 13,428.8 / 9,311.2 / 7,367.6 / 5,305.9) the area
    regexes were built to tolerate;
  * one deliberately-wrong synthetic per task (a misread that flips a threshold / moves the winner)
    -> the keystone check scores 0, so the composer cannot launder a bad extraction into credit.

``observability`` carries a synthetic ``{"visit": {"count": N}}`` because every keystone here is
grounding-gated (a value string alone must never earn credit); the real run's visits happen in the
leaves, upstream of and independent of composition.
"""
from agent.app.testing import execution_compiled as ec
from agent.app.idea_tests import test_072_tier5_count_with_condition as t072
from agent.app.idea_tests import test_076_tier5_numeric_and_filter as t076
from agent.app.idea_tests import test_078_tier5_count_with_condition_b as t078


_WIKI = "https://en.wikipedia.org/wiki/"


def _fact(value, name):
    """One leaf's resolved fact, exactly as ``_run_leaf_thin`` returns it."""
    return f"{value} — source: {_WIKI}{name.replace(' ', '_')}"


def _num(value):
    """The figure as a leaf would report it off an infobox (no trailing '.0')."""
    return f"{int(value)}" if float(value).is_integer() else f"{value}"


def _compose(module, results):
    plan = module.get_compiled_plan()
    return ec._compose(plan["leaves"], results, plan["composition"])


def _score_all(module, composed, n_visits):
    """Every real validator's score for a composed deliverable, keyed by check name."""
    result = {"output": {"final_deliverable": composed}}
    observability = {"visit": {"count": n_visits}}
    verdicts = [check(result, observability) for check in module.get_validation_functions()]
    return {v["check"]: v for v in verdicts}


# ── 072: count_threshold over seven lake depths ───────────────────────────────────────────────────

def _results_072(overrides=None):
    overrides = overrides or {}
    return {
        f"{e['key']}_depth": _fact(f"{overrides.get(e['key'], e['depth'])} m", e["name"])
        for e in t072.ENTITIES
    }


def test_072_composed_answer_passes_every_real_validator():
    composed = _compose(t072, _results_072())
    assert composed is not None
    scores = _score_all(t072, composed, len(t072.ENTITIES))
    assert all(v["score"] == 1.0 for v in scores.values()), scores
    # The keystone count is the composed one, not a coincidence of the fixture text.
    assert composed.startswith(f"{t072.KEYSTONE_COUNT} of the {len(t072.ENTITIES)} lakes")


def test_072_misread_depth_flips_the_count_and_fails_the_keystone():
    """Tinnsjå (460 m) misread as 560 m -> count 5. The keystone must score 0: composition removes
    the LLM's counting mistakes, it does not launder a bad extraction."""
    composed = _compose(t072, _results_072({"tinnsja": 560}))
    assert composed.startswith("5 of the 7 lakes")
    scores = _score_all(t072, composed, len(t072.ENTITIES))
    assert scores["keystone_count"]["score"] == 0.0
    # ... and the gated secondaries short-circuit with it.
    assert scores["passing_lakes"]["score"] == 0.0
    assert scores["citation"]["score"] == 0.0


def test_072_unresolved_leaf_declines_rather_than_undercounting():
    results = _results_072()
    results["sarez_depth"] = f"UNKNOWN — {_WIKI}Sarez_Lake"
    assert _compose(t072, results) is None


# ── 078: count_threshold over seven island areas (decimal-heavy) ──────────────────────────────────

# The REAL infobox strings — the rounded integers in ENTITIES are what the regexes tolerate, not
# what a leaf actually reads off the page.
_078_INFOBOX = {
    "bangka": "11,831.02", "chichagof": "5,305.9", "flores": "14,250", "kodiak": "9,311.2",
    "leyte": "7,367.6", "melville": "5,786", "samar": "13,428.8",
}


def _results_078(raw=None, overrides=None):
    raw = raw if raw is not None else {e["key"]: f"{e['area']:,}" for e in t078.ENTITIES}
    overrides = overrides or {}
    return {
        f"{e['key']}_area": _fact(f"{overrides.get(e['key'], raw[e['key']])} km²", e["name"])
        for e in t078.ENTITIES
    }


def test_078_composed_answer_passes_every_real_validator():
    composed = _compose(t078, _results_078())
    assert composed is not None
    scores = _score_all(t078, composed, len(t078.ENTITIES))
    assert all(v["score"] == 1.0 for v in scores.values()), scores
    assert composed.startswith(f"{t078.KEYSTONE_COUNT} of the {len(t078.ENTITIES)} islands")


def test_078_unrounded_infobox_figures_still_match_every_area_regex():
    """The area regexes end in a digit CLASS (e.g. ``11[,\\s]?83[0-9]``) precisely to tolerate the
    rounding ambiguity; '(?!\\d)' is satisfied by the decimal point, so the unrounded render matches."""
    composed = _compose(t078, _results_078(raw=_078_INFOBOX))
    assert "11,831.02" in composed and "7,367.6" in composed
    scores = _score_all(t078, composed, len(t078.ENTITIES))
    assert all(v["score"] == 1.0 for v in scores.values()), scores


def test_078_misread_area_flips_the_count_and_fails_the_keystone():
    """Leyte (7,368 km²) misread as 9,368 km² -> count 5 -> keystone 0."""
    composed = _compose(t078, _results_078(overrides={"leyte": "9,368"}))
    assert composed.startswith("5 of the 7 islands")
    scores = _score_all(t078, composed, len(t078.ENTITIES))
    assert scores["keystone_count"]["score"] == 0.0
    assert scores["passing_islands"]["score"] == 0.0


# ── 076: and_filter over six lakes x two constraints ──────────────────────────────────────────────

def _results_076(depth_overrides=None):
    depth_overrides = depth_overrides or {}
    results = {}
    for e in t076.ENTITIES:
        results[f"{e['key']}_area"] = _fact(f"{e['area']:,} km²", e["name"])
        depth = depth_overrides.get(e["key"], e["depth"])
        results[f"{e['key']}_depth"] = _fact(f"{_num(depth)} m", e["name"])
    return results


def test_076_composed_answer_passes_every_real_validator():
    composed = _compose(t076, _results_076())
    assert composed is not None
    scores = _score_all(t076, composed, len(t076.ENTITIES))
    assert all(v["score"] == 1.0 for v in scores.values()), scores
    # The unique satisfier is named in the lead sentence, by the fixture's own derived winner.
    assert composed.startswith(f"{t076.WINNER['name']} is the unique lake")


def test_076_composed_answer_satisfies_both_keystone_paths():
    """The render is accepted by the explicit-assertion path AND, independently, by the
    enumeration path — so a future tightening of either one cannot silently zero the task."""
    composed = _compose(t076, _results_076())
    stripped = t076._strip_urls(composed)
    assert t076._WINNIPEGOSIS_WINS.search(stripped)      # path 2: 'unique' next to the winner
    assert t076._keystone_enumeration_ok(stripped)       # path 3: the winner's row, 2x 'yes'
    assert not t076._OTHER_WINS.search(stripped)         # no decoy is asserted as the answer


def test_076_swapped_depths_move_the_winner_and_fail_the_keystone():
    """Winnipegosis misread as 120 m and Athabasca as 12 m -> Athabasca becomes the unique
    satisfier. The keystone must score 0 (and the gated secondaries with it)."""
    composed = _compose(t076, _results_076({"winnipegosis": 120.0, "athabasca": 12.0}))
    assert composed.startswith("Lake Athabasca is the unique lake")
    scores = _score_all(t076, composed, len(t076.ENTITIES))
    assert scores["keystone_filter"]["score"] == 0.0
    assert scores["winner_attributes"]["score"] == 0.0
    assert scores["citation"]["score"] == 0.0


def test_076_ambiguous_extraction_declines_instead_of_guessing():
    """Two satisfiers (a second lake misread shallow) and zero satisfiers both fall back — the
    uniqueness claim is unsupportable in either case."""
    assert _compose(t076, _results_076({"nettilling": 15.0})) is None
    assert _compose(t076, _results_076({"winnipegosis": 120.0})) is None


def test_076_unresolved_leaf_declines_rather_than_half_filtering():
    results = _results_076()
    results["khanka_depth"] = "UNKNOWN"
    assert _compose(t076, results) is None
