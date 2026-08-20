"""Tests for deterministic (LLM-free) value threading — ``idea_policies/waypoint.py``.

Three layers, matching ``contract_reexpand_test.py``'s split:

  * the pure ``_number_near``/``_find_number_near`` refactor is a byte-identical regression
    lock for ``evaluate_step_contract`` (F33, the contract log) — the refactor MUST NOT move
    that output, independent of the cue-vocabulary widening riding along with it;
  * the pure ``build_waypoint`` extractor: the page-identity guard (the precision measurement
    that gates this design — ``scripts/replay_waypoints.py --precision`` found the cue-window
    and anchor-text extractors return a value on essentially 100% of wrong-page trials, and
    that value is wrong essentially 100% of the time), the two extractor paths, and every
    "never emit a bare value" invariant;
  * the engine wiring at the single shared completion point in ``_handle_action_result``:
    flag OFF is byte-identical, flag ON attaches ``details["waypoint"]`` for a VISIT leaf.
"""
from __future__ import annotations

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies import BestScoreSelectionPolicy, SimpleMergePolicy
from agent.app.idea_policies.base import (
    DetailKey,
    ExpansionPolicy,
    EvaluationPolicy,
    DecompositionPolicy,
    IdeaActionType,
    IdeaNodeStatus,
)
from agent.app.idea_policies.actions import LeafAction
from agent.app.idea_policies.contract_satisfaction import (
    _DIGIT_RE,
    _QUANTITY_CUES,
    _find_number_near,
    _number_near,
    evaluate_step_contract,
)
from agent.app.idea_policies.waypoint import Waypoint, build_waypoint


# ------------------------------------------------------------------------------------------
# layer 1 — the _number_near -> _find_number_near refactor must not move evaluate_step_contract
# ------------------------------------------------------------------------------------------


class _Node:
    """Minimal duck-typed node (mirrors contract_reexpand_test._Node)."""

    def __init__(self, title, details):
        self.title = title
        self.details = details


def _visit_node(content, *, goal="Find the maximum depth of Quesnel Lake", **details):
    base = {
        DetailKey.ACTION.value: IdeaActionType.VISIT.value,
        DetailKey.GOAL.value: goal,
        DetailKey.ACTION_RESULT.value: {
            "success": True, "url": "https://en.wikipedia.org/wiki/Quesnel_Lake",
            "title": "Quesnel Lake", "content": content,
        },
    }
    base.update(details)
    return _Node(goal, base)


def test_find_number_near_is_presence_equivalent_to_the_original_digit_search():
    """The refactor's own correctness argument, executable: `_find_number_near` searches with
    a fuller "whole number" regex instead of the original single-digit `_DIGIT_RE`, but both
    start matching at the same first-digit position, so *whether* a match exists is identical
    for any window -- only what a match's `.group()` contains differs (the whole number instead
    of one digit character)."""
    samples = [
        "no digits here at all",
        "the depth is 511 m below the surface",
        "born in 1869, died before completion",
        "",
        "12345",
        "a b c 7",
    ]
    for window_text in samples:
        assert (_find_number_near(window_text, ()) is not None) == bool(_DIGIT_RE.search(window_text))


def test_number_near_bool_wrapper_is_unchanged_by_the_refactor():
    evidence_low = "quesnel lake maximum depth: 511 m below the surface, fed by the horsefly river."
    for cue, canonical, variants in _QUANTITY_CUES:
        assert _number_near(evidence_low, variants) == (_find_number_near(evidence_low, variants) is not None)
    # And the "no stable wording" (empty-variants) branch, used by year/count/date cues.
    assert _number_near(evidence_low, ()) == (_find_number_near(evidence_low, ()) is not None)
    assert _number_near("no numbers anywhere in this sentence", ()) is False


def test_find_number_near_picks_the_value_nearest_the_cue_not_the_leftmost_in_window():
    """Regression lock for the 2026-08-16 selection-rule fix: a `re.search` on a ±200-char
    window returns the LEFTMOST number in read order, not the one nearest the cue -- corpus
    case (task 136/keystone, replayed via ``scripts/replay_waypoints.py --oracle``): the ship's
    real length sits 4 chars after "length", but an unrelated date sitting earlier in the same
    window used to win. Pinned here word-for-word against that real page text."""
    evidence_low = (
        "struck rocks on 27 august 1862. no larger ship in all respects until 1901 by the rms "
        "celtic. general characteristics type passenger ship tonnage 18,915 grt, 13,344 nrt "
        "displacement 32,160 tons length 692 ft (211 m) beam 82 ft (25 m) decks 4 decks"
    )
    match = _find_number_near(evidence_low, ("length", "long"))
    assert match is not None
    assert match.group(0) == "692", (
        f"expected the value 4 chars after the cue (692), got the leftmost-in-window "
        f"distractor instead: {match.group(0)!r}"
    )


def test_find_number_near_prefers_a_closer_occurrence_of_a_later_variant():
    """Nearest is GLOBAL across every occurrence of every variant, not "first occurrence that
    has any hit at all": variant 1 ("elevation") is tried first and has a hit (a FAR distractor
    within its own window), but variant 2 ("altitude") has a much CLOSER hit -- the closer one
    must win even though it comes from the later variant."""
    evidence_low = (
        "elevation " + ("x" * 100) + " 9999 distractor reading, " + ("x" * 60) +
        "altitude: 511 m above sea level."
    )
    match = _find_number_near(evidence_low, ("elevation", "altitude"))
    assert match is not None and match.group(0) == "511"


def test_find_number_near_multiple_cue_occurrences_takes_the_globally_nearest():
    """Two occurrences of the SAME variant, each with its own nearby candidate -- the globally
    nearest number to ANY occurrence wins, not just the first occurrence's own local pick (which
    is exactly what the pre-fix "leftmost in the first hit's window" behavior returned)."""
    evidence_low = (
        "span " + ("z" * 50) + " 999 distractor. " + ("w" * 150) +
        "main span of 1,057 feet exactly."
    )
    match = _find_number_near(evidence_low, ("span", "main span"))
    assert match is not None
    # "999" sits close to the FIRST "span" occurrence; "1,057" sits closer still to the SECOND
    # ("main span") occurrence -- the true value must win the global nearest-of-all comparison.
    assert match.group(0) == "1,057"


def test_evaluate_step_contract_unchanged_for_pre_existing_cue_vocabulary():
    """Regression lock: for contract text built only from cues that existed BEFORE this
    change (i.e. independent of the "span" vocabulary widening riding along with the
    refactor), `evaluate_step_contract`'s verdict is exactly what it was -- F33 and the
    contract log must not move."""
    right_page = (
        "Quesnel Lake is a lake in British Columbia. Maximum depth: 511 m. "
        "Surface elevation 726 m."
    )
    wrong_page = "Quesnel is a city in the Cariboo region of British Columbia known for forestry."
    mandate = "Report the maximum depth of Quesnel Lake in metres. Do not guess."

    satisfied = evaluate_step_contract(_visit_node(right_page), mandate)
    assert (satisfied.applicable, satisfied.satisfied) == (True, True)

    unsatisfied = evaluate_step_contract(
        _visit_node(wrong_page, **{DetailKey.ACTION_RESULT.value: {
            "success": True, "url": "https://en.wikipedia.org/wiki/Quesnel",
            "title": "Quesnel", "content": wrong_page,
        }}),
        mandate,
    )
    assert (unsatisfied.applicable, unsatisfied.satisfied) == (True, False)
    assert any("depth" in m for m in unsatisfied.missing)


def test_span_cue_widens_required_datums_without_touching_unrelated_text():
    """The vocabulary widening (explicitly asked for by the audit: "span" was a gap) is a
    deliberate, disclosed behavior change -- distinct from the refactor's own byte-identical
    guarantee above -- so it gets its own positive test rather than being folded into the
    regression lock."""
    from agent.app.idea_policies.contract_satisfaction import _required_datums
    assert ("span", ()) not in _required_datums("Find the lake's depth")  # sanity: unrelated text unaffected
    assert any(canon == "span" for canon, _v in _required_datums("The terminal bridge's main span"))


# ------------------------------------------------------------------------------------------
# layer 2 — build_waypoint: page-identity guard + the two extractor paths
# ------------------------------------------------------------------------------------------


class _WpNode:
    def __init__(self, details):
        self.details = details
        self.title = details.get(DetailKey.GOAL.value, "")


_CINCINNATI_CONTRACT_GOAL = "The terminal bridge's main span — source URL"
_CINCINNATI_PARENT_GOAL = (
    "Open the Wikipedia page of the bridge identified in the previous step (John A. Roebling "
    "Suspension Bridge, Cincinnati) and read its main span."
)


def _cincinnati_node(**extra_details):
    details = {DetailKey.GOAL.value: _CINCINNATI_CONTRACT_GOAL, DetailKey.PARENT_GOAL.value: _CINCINNATI_PARENT_GOAL}
    details.update(extra_details)
    return _WpNode(details)


def _cincinnati_correct_result():
    return {
        "url": "https://en.wikipedia.org/wiki/John_A._Roebling_Suspension_Bridge",
        "page_title": "John A. Roebling Suspension Bridge",
        "content_full": (
            "The John A. Roebling Suspension Bridge has a main span of 1,057 feet (322 m), "
            "the longest in the world when it opened in 1867."
        ),
    }


def _brooklyn_wrong_result():
    """The audit's concrete example: the same waypoint's cue-window path pointed at the
    Brooklyn Bridge (a real page from the same run, NOT the correct source)."""
    return {
        "url": "https://en.wikipedia.org/wiki/Brooklyn_Bridge",
        "page_title": "Brooklyn Bridge",
        "content_full": "The Brooklyn Bridge has a main span of 1,595 feet (486 m).",
    }


def test_cue_window_extracts_the_value_on_the_correct_page():
    wp = build_waypoint(_cincinnati_node(), _cincinnati_correct_result(), "visit", "n1", 3)
    assert wp.method == "cue_window"
    assert wp.value == "1,057"
    assert wp.value_kind == "span"
    assert wp.source_url == "https://en.wikipedia.org/wiki/John_A._Roebling_Suspension_Bridge"
    assert "1,057" in wp.evidence
    assert wp.node_id == "n1" and wp.step == 3


def test_cue_window_end_to_end_nearest_selection_on_real_infobox_text():
    """End-to-end regression lock (through ``build_waypoint``, not just ``_find_number_near``
    directly) for the 2026-08-16 selection fix, using the real SS Great Eastern corpus text
    (``scripts/replay_waypoints.py --oracle``, task 136/keystone) where the OLD leftmost-in-
    window pick returned "27" (from "struck rocks on 27 august 1862", read earlier in the same
    ±200-char window) instead of the correct length, 692 ft, sitting 4 chars after the cue."""
    node = _WpNode({DetailKey.GOAL.value: "Read the SS Great Eastern's total length from its Wikipedia page."})
    result = {
        "url": "https://en.wikipedia.org/wiki/SS_Great_Eastern",
        "page_title": "SS Great Eastern",
        "content_full": (
            "Struck rocks on 27 August 1862. No larger ship in all respects until 1901 by the "
            "RMS Celtic. General characteristics Type passenger ship Tonnage 18,915 GRT, "
            "13,344 NRT Displacement 32,160 tons Length 692 ft (211 m) Beam 82 ft (25 m) "
            "Decks 4 decks"
        ),
    }
    wp = build_waypoint(node, result, "visit", "n1b", 1)
    assert wp.method == "cue_window"
    assert wp.value == "692"


def test_page_identity_guard_blocks_the_wrong_page_confident_lie():
    """The precision finding this whole guard exists because of, pinned as a test: applied
    to a WRONG page that nonetheless has its own "span" figure, the extractor must abstain
    (method="none", value=None) rather than confidently return the Brooklyn Bridge's span as
    if it were the Cincinnati bridge's."""
    wp = build_waypoint(_cincinnati_node(), _brooklyn_wrong_result(), "visit", "n2", 3)
    assert wp.method == "none"
    assert wp.value is None
    assert wp.value_kind is None
    assert wp.evidence == ""
    # source_url/source_title are still populated for offline diagnosis, per the "method
    # explains failures" contract -- only the (unverified) VALUE is withheld.
    assert wp.source_url == "https://en.wikipedia.org/wiki/Brooklyn_Bridge"


def test_guard_fails_closed_when_the_contract_carries_no_subject_tokens_at_all():
    """A boilerplate leaf goal (no proper noun anywhere, and no PARENT_GOAL to fall back to)
    must NOT vacuously pass the guard -- unlike `_subject_present`'s "zero tokens is trivially
    satisfied" convention, which would defeat the guard on exactly this leaf."""
    node = _WpNode({DetailKey.GOAL.value: "Visit a source page to substantiate the assigned research goal"})
    wp = build_waypoint(node, _cincinnati_correct_result(), "visit", "n3", 1)
    assert wp.method == "none"
    assert wp.value is None


def test_generic_template_goal_borrows_the_subject_but_not_a_datum_cue_from_the_parent():
    """The audit's other flagged gap: an injected leaf's own GOAL is boilerplate (no subject,
    no datum). Its PARENT_GOAL (already denormalized onto every expanded child) DOES carry a
    subject, so it widens the identity guard -- letting a genuinely correct page pass instead
    of failing closed. It deliberately does NOT also borrow a datum cue from the parent: a
    corpus-driven finding is pinned right below this test (parent/mandate text routinely
    describes OTHER hops too, e.g. a "span" ask that belongs to a later hop in the same chain,
    and borrowing it here would make an entity-only leaf go hunting for an unrelated number).
    So this leaf -- genuinely asking for nothing measurable -- stays method="none" even on the
    correct page, rather than fabricate a value its own contract never asked for."""
    node = _WpNode({
        DetailKey.GOAL.value: "Visit a source page to substantiate the assigned research goal",
        DetailKey.PARENT_GOAL.value: _CINCINNATI_PARENT_GOAL,
    })
    wp = build_waypoint(node, _cincinnati_correct_result(), "visit", "n4", 1)
    assert wp.method == "none"
    assert wp.value is None


def test_parent_goal_datum_cues_are_never_borrowed_even_when_the_leaf_has_none():
    """Corpus-driven regression: widening DATUM cues (not just subject tokens) from
    PARENT_GOAL was the first design tried here, and it broke a real "creator" (entity, no
    number to find) leaf from the audit corpus -- its own contract text asked for nothing
    measurable, but the immediate parent step described the WHOLE multi-hop chain, including a
    later hop's "span"/"length" ask, so the widened cue fired on an unrelated number elsewhere
    on the (correct) page. An entity leaf whose own text has no datum must stay method="none"
    (or fall through to anchor-text) -- never borrow a number-shaped ask from an ancestor."""
    node = _WpNode({
        DetailKey.GOAL.value: "Visit the Clifton Suspension Bridge Wikipedia page to identify the engineer.",
        DetailKey.PARENT_GOAL.value: (
            "HOP 1 -- identify the engineer of the Clifton Suspension Bridge. HOP 3 -- read "
            "the terminal ship's length in feet from its Wikipedia page."
        ),
    })
    result = {
        "url": "https://en.wikipedia.org/wiki/Clifton_Suspension_Bridge",
        "page_title": "Clifton Suspension Bridge",
        "content_full": (
            "The Clifton Suspension Bridge is 51.4549 degrees north. It was designed by "
            "Isambard Kingdom Brunel."
        ),
        "link_contexts": {
            "https://en.wikipedia.org/wiki/Isambard_Kingdom_Brunel": "Isambard Kingdom Brunel",
        },
    }
    wp = build_waypoint(node, result, "visit", "n4b", 1)
    assert wp.method != "cue_window", "must not borrow the later hop's 'length' cue"
    assert wp.method == "anchor_text"
    assert wp.value == "Isambard Kingdom Brunel"


def test_anchor_text_extracts_an_entity_via_link_contexts():
    # Corpus-matched phrasing (real 135/creator leaf goals never mention "died" -- that word
    # was in the STATIC compiled-plan instruction text, not what any observed executed leaf's
    # own goal actually says; see the sibling gate tests below for what happens when a cue word
    # like "died"/"born" DOES leak into an entity leaf's own text).
    node = _WpNode({
        DetailKey.GOAL.value: "Open the Wikipedia page for the Brooklyn Bridge to find its chief engineer.",
    })
    result = {
        "url": "https://en.wikipedia.org/wiki/Brooklyn_Bridge",
        "page_title": "Brooklyn Bridge",
        "content_full": "The bridge's designer was John A. Roebling, who died before it was built.",
        "link_contexts": {
            "https://en.wikipedia.org/wiki/John_A._Roebling": "John A. Roebling",
            "https://en.wikipedia.org/wiki/Washington_Roebling": "Washington Roebling",
            "https://en.wikipedia.org/wiki/Main_Page": "Main page",
        },
    }
    wp = build_waypoint(node, result, "visit", "n5", 1)
    assert wp.method == "anchor_text"
    assert wp.value == "John A. Roebling"
    assert wp.value_kind == "anchor"


def test_anchor_text_gate_can_false_positive_on_an_entity_leaf_that_happens_to_say_died():
    """Documented trade-off, not silently papered over: `_QUANTITY_CUES` maps bare "died" to a
    (no-stable-wording) year cue. An entity leaf whose OWN phrasing happens to use that word
    (e.g. "...its designer, who died before completion") gets its datums list non-empty and the
    anchor_text gate blocks it -- even though the ask is really "who", not "when". Measured
    against the real corpus (scripts/replay_waypoints.py --validate-product): this false
    positive fired on exactly 1 of 74 real correct-page trials where anchor_text would
    otherwise have run (a 065/town leaf whose goal also asked for elevation) -- a real but rare
    cost of a gate that otherwise removes a class of impossible answers for free."""
    node = _WpNode({
        DetailKey.GOAL.value: "Open the Wikipedia page for the Brooklyn Bridge. Its designer "
                              "died before the bridge was completed.",
    })
    result = {
        "url": "https://en.wikipedia.org/wiki/Brooklyn_Bridge",
        "page_title": "Brooklyn Bridge",
        "content_full": "The bridge's designer was John A. Roebling, who died before it was built.",
        "link_contexts": {
            "https://en.wikipedia.org/wiki/John_A._Roebling": "John A. Roebling",
        },
    }
    wp = build_waypoint(node, result, "visit", "n5d", 1)
    assert wp.method == "none"


def test_anchor_text_gated_off_for_a_leaf_whose_own_contract_asks_for_a_quantity():
    """Follow-up measurement (coordinator, second pass): every anchor_text emission observed
    against a leaf whose own contract carried a datum cue (a quantity/date ask -- confirmed via
    `own_contract.datums`) was wrong by construction: an anchor can never legitimately answer
    "what is the height". cue_window fails to find a number here on purpose (isolating the
    gate); there must be no anchor_text fallback even though a plausible-looking anchor exists
    in the window of a DIFFERENT (unrelated) cue."""
    node = _WpNode({
        DetailKey.GOAL.value: "Visit the Saturn V Wikipedia page to find its height.",
    })
    result = {
        "url": "https://en.wikipedia.org/wiki/Saturn_V",
        "page_title": "Saturn V",
        "content_full": "Designed by Wernher von Braun's team, the Saturn V was a launch vehicle.",
        "link_contexts": {
            "https://en.wikipedia.org/wiki/Wernher_von_Braun": "Wernher von Braun",
        },
    }
    wp = build_waypoint(node, result, "visit", "n5b", 1)
    assert wp.method == "none"
    assert wp.value is None


def test_anchor_text_still_fires_for_a_leaf_with_no_datum_cue_at_all():
    """The gate is scoped to "own contract asked for a number", not "a number exists
    anywhere nearby" -- an entity-only leaf (no datums) must still reach anchor_text."""
    node = _WpNode({
        DetailKey.GOAL.value: "Visit the Saturn V Wikipedia page to find who designed it.",
    })
    result = {
        "url": "https://en.wikipedia.org/wiki/Saturn_V",
        "page_title": "Saturn V",
        "content_full": "Designed by Wernher von Braun's team, the Saturn V was a launch vehicle.",
        "link_contexts": {
            "https://en.wikipedia.org/wiki/Wernher_von_Braun": "Wernher von Braun",
        },
    }
    wp = build_waypoint(node, result, "visit", "n5c", 1)
    assert wp.method == "anchor_text"
    assert wp.value == "Wernher von Braun"


def test_no_value_never_carries_evidence_or_a_kind():
    """"Never emit a bare value" also means the inverse: no method means no evidence either."""
    node = _cincinnati_node()
    result = dict(_cincinnati_correct_result())
    result["content_full"] = "This page mentions nothing measurable at all."
    wp = build_waypoint(node, result, "visit", "n6", 1)
    assert wp.method == "none"
    assert wp.value is None and wp.value_kind is None and wp.evidence == ""
    assert wp.source_url  # still populated, for diagnosis


def test_empty_content_is_method_none_not_a_crash():
    wp = build_waypoint(_cincinnati_node(), {"url": "https://x", "content_full": ""}, "visit", "n7", 1)
    assert wp.method == "none"
    assert wp.value is None


@pytest.mark.parametrize("action", ["search", "think", "save", "merge", None])
def test_non_visit_actions_produce_no_waypoint_at_all(action):
    wp = build_waypoint(_cincinnati_node(), _cincinnati_correct_result(), action, "n8", 1)
    assert wp is None


def test_non_dict_result_produces_no_waypoint():
    assert build_waypoint(_cincinnati_node(), None, "visit", "n9", 1) is None
    assert build_waypoint(_cincinnati_node(), "not a dict", "visit", "n9", 1) is None


def test_waypoint_as_dict_round_trips_every_field():
    wp = build_waypoint(_cincinnati_node(), _cincinnati_correct_result(), "visit", "n10", 5)
    d = wp.as_dict()
    assert set(d) == {
        "hop_goal", "value", "value_kind", "evidence", "source_url", "source_title",
        "node_id", "step", "method",
    }
    assert d["node_id"] == "n10" and d["step"] == 5


# ------------------------------------------------------------------------------------------
# layer 3 — engine wiring at the shared completion point in _handle_action_result
# ------------------------------------------------------------------------------------------


class DummyIO:
    def set_telemetry(self, telemetry):
        return None


class FakeExpansion(ExpansionPolicy):
    def __init__(self, settings=None):
        super().__init__(settings=settings)
        self.calls = 0

    async def expand(self, graph: IdeaDag, node_id: str, memories=None):
        self.calls += 1
        return [{
            "title": "follow-up", "details": {DetailKey.ACTION.value: IdeaActionType.THINK.value},
            "score": None,
        }]


class FakeEvaluation(EvaluationPolicy):
    async def evaluate(self, graph, node_id):
        graph.evaluate(node_id, 0.6)
        return 0.6

    async def evaluate_batch(self, graph, parent_id, candidate_ids):
        return {nid: (graph.evaluate(nid, 0.6) or 0.6) for nid in candidate_ids}


class FakeDecomposition(DecompositionPolicy):
    def should_decompose(self, graph, node_id):
        return False


class VisitAction(LeafAction):
    RESULT = None

    async def execute(self, graph, node_id, io):
        return dict(VisitAction.RESULT)


class FakeRegistry:
    def __init__(self, settings):
        self.settings = dict(settings or {})

    def get(self, action_type):
        return VisitAction(settings=self.settings)


def _make_engine(*, waypoint_enabled):
    settings = {
        "allow_unscored_selection": True,
        "min_score_threshold": 0.0,
        "best_first_global": False,
        "got_contract_reexpand_enabled": False,
        "got_step_confidence_judge_enabled": False,
        "got_step_confidence_reexpand_enabled": False,
        "got_reexpand_enabled": False,
        "got_dedup_enabled": False,
        "got_embed_on_create": False,
        "auto_parallel_siblings": False,
        "waypoint_enabled": waypoint_enabled,
    }
    expansion = FakeExpansion(settings)
    engine = IdeaDagEngine(
        io=DummyIO(),
        settings=settings,
        expansion=expansion,
        evaluation=FakeEvaluation(settings),
        selection=BestScoreSelectionPolicy(settings=settings),
        decomposition=FakeDecomposition(settings),
        merge=SimpleMergePolicy(settings=settings),
        actions=FakeRegistry(settings),
    )
    return engine, expansion


def _graph_with_cincinnati_leaf(result):
    graph = IdeaDag(root_title="root")
    root = graph.get_node(graph.root_id())
    root.details["mandate"] = "Report the terminal bridge's main span. Do not guess."
    parent = graph.add_child(graph.root_id(), "parent goal", details={
        DetailKey.GOAL.value: _CINCINNATI_PARENT_GOAL,
    })
    leaf = graph.add_child(parent.node_id, _CINCINNATI_CONTRACT_GOAL, details={
        DetailKey.ACTION.value: IdeaActionType.VISIT.value,
        DetailKey.GOAL.value: _CINCINNATI_CONTRACT_GOAL,
        DetailKey.PARENT_GOAL.value: _CINCINNATI_PARENT_GOAL,
        DetailKey.IS_LEAF.value: True,
    })
    VisitAction.RESULT = {"action": IdeaActionType.VISIT.value, **result}
    return graph, parent, leaf


@pytest.mark.asyncio
async def test_flag_off_writes_no_waypoint_and_is_otherwise_unaffected():
    engine, expansion = _make_engine(waypoint_enabled=False)
    graph, parent, leaf = _graph_with_cincinnati_leaf({
        "success": True, **_cincinnati_correct_result(),
        "content": _cincinnati_correct_result()["content_full"],
    })

    result = await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert leaf.status == IdeaNodeStatus.DONE
    assert result == parent.node_id
    assert "waypoint" not in leaf.details
    assert expansion.calls == 0


@pytest.mark.asyncio
async def test_flag_on_attaches_a_waypoint_for_a_visit_leaf_on_the_correct_page():
    engine, expansion = _make_engine(waypoint_enabled=True)
    graph, parent, leaf = _graph_with_cincinnati_leaf({
        "success": True, **_cincinnati_correct_result(),
        "content": _cincinnati_correct_result()["content_full"],
    })

    await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert leaf.status == IdeaNodeStatus.DONE
    wp = leaf.details.get("waypoint")
    assert wp is not None
    assert wp["method"] == "cue_window"
    assert wp["value"] == "1,057"
    assert wp["node_id"] == leaf.node_id


@pytest.mark.asyncio
async def test_flag_on_still_completes_the_leaf_when_extraction_abstains():
    """The wrong-page case must not fail the leaf -- it completes exactly as it always did,
    just with `method="none"` recorded for offline diagnosis."""
    engine, expansion = _make_engine(waypoint_enabled=True)
    wrong = _brooklyn_wrong_result()
    graph, parent, leaf = _graph_with_cincinnati_leaf({
        "success": True, **wrong, "content": wrong["content_full"],
    })

    result = await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert leaf.status == IdeaNodeStatus.DONE
    assert result == parent.node_id
    wp = leaf.details.get("waypoint")
    assert wp is not None
    assert wp["method"] == "none"
    assert wp["value"] is None


@pytest.mark.asyncio
async def test_flag_on_extraction_exception_does_not_fail_the_leaf(monkeypatch):
    """Defense in depth: even if the extractor itself misbehaves, the leaf still completes."""
    import agent.app.idea_policies.waypoint as waypoint_mod

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(waypoint_mod, "build_waypoint", _boom)
    engine, expansion = _make_engine(waypoint_enabled=True)
    graph, parent, leaf = _graph_with_cincinnati_leaf({
        "success": True, **_cincinnati_correct_result(),
        "content": _cincinnati_correct_result()["content_full"],
    })

    result = await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert leaf.status == IdeaNodeStatus.DONE
    assert result == parent.node_id
    assert "waypoint" not in leaf.details
