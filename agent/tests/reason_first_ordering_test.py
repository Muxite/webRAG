"""Reason-before-answer field ordering at three call sites (opt-in, all default OFF).

promptbench v2 (``docs/handoffs/PROMPTBENCH_V2_RESULTS_2026-08-19.md``) measured, pooled
over 5 models with a cluster bootstrap over task modules:

  * verify   ``A2 - A1`` = +0.142, CI [+0.053, +0.232], p = 0.0119, positive on 5/5
  * followup ``A2 - A1`` = +0.196, CI [+0.080, +0.312], p = 0.0064 -- the largest effect
  * merge ``goal_achieved``: the SHIPPED ordering is degenerate on 5/5 models, each
    answering ACHIEVED on 100% of a balanced set. No paired contrast is computable
    against a constant, so that site's evidence is defect-grade rather than an effect.

The pre-registered gate authorizes implementing these flagged and default OFF. It does
NOT authorize changing a shipped prompt, and ``promptbench_shipped_parity_test`` holds
the line: each flag selects a separate variant constant, and the shipped text is
byte-untouched.

Two failure modes are pinned deliberately, because both would look like success:

  * a flag that is wired but whose variant is identical to shipped (an invisible no-op);
  * a merge reorder applied to only ONE of merge's TWO ordering sources, leaving the
    model reading two conflicting orders in the same system message.

No network: the LLM call is a scripted response.
"""
from __future__ import annotations

import asyncio
import json

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_schemas import (
    DEFAULT_JSON_SCHEMAS,
    MERGE_JSON_SCHEMA,
    MERGE_JSON_SCHEMA_GOAL_EVAL_FIRST,
)
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_policies.actions import MergeLeafAction, VerifyLeafAction
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus
from agent.app.got_operations import _FOLLOWUP_REASON_FIRST_SYSTEM_PROMPT


SETTINGS = load_idea_dag_settings()

_MERGE_RESPONSE = json.dumps({
    "summary": "s", "key_findings": ["k"], "goal_achieved": True,
    "goal_evaluation": "done", "missing_requirements": [],
})
_VERIFY_RESPONSE = json.dumps({
    "verdict": "TRUE", "confidence": 0.9, "supporting_url": "https://e.example",
    "contradicting_url": "", "quote": "q", "reasoning": "r",
})


class _RecordingIO:
    """Captures the payload the action actually built, then returns a scripted reply."""

    def __init__(self, response):
        self._response = response
        self.payloads = []

    def build_llm_payload(self, messages=None, **kw):
        payload = {"messages": messages, **kw}
        self.payloads.append(payload)
        return payload

    async def query_llm_with_fallback(self, payload, model_name=None, fallback_model=None,
                                      timeout_seconds=None):
        return self._response

    def _system(self):
        return self.payloads[-1]["messages"][0]["content"]


def _settings(**overrides):
    s = load_idea_dag_settings()
    s.update(overrides)
    return s


def _merge_graph() -> IdeaDag:
    g = IdeaDag(root_title="root")
    node = g.add_child(g.root_id(), "merge me", status=IdeaNodeStatus.PENDING)
    node.details[DetailKey.MERGED_RESULTS.value] = [
        {"title": "a", "content": "alpha"}, {"title": "b", "content": "beta"},
    ]
    return g, node.node_id


def _run_merge(**overrides):
    g, node_id = _merge_graph()
    io = _RecordingIO(_MERGE_RESPONSE)
    action = MergeLeafAction(settings=_settings(**overrides))
    result = asyncio.run(action.execute(g, node_id, io))
    return io, result


def _verify_graph() -> IdeaDag:
    g = IdeaDag(root_title="root")
    # verify refuses to run without gathered evidence, so give it a completed sibling visit.
    g.add_child(
        g.root_id(), "visit the source",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.ACTION_RESULT.value: {
                "success": True, "action": IdeaActionType.VISIT.value,
                "url": "https://e.example", "content": "The claim is stated here.",
            },
        },
        status=IdeaNodeStatus.DONE,
    )
    node = g.add_child(
        g.root_id(), "verify me",
        details={DetailKey.ACTION.value: IdeaActionType.VERIFY.value, "claim": "the claim"},
        status=IdeaNodeStatus.PENDING,
    )
    return g, node.node_id


def _run_verify(**overrides):
    g, node_id = _verify_graph()
    io = _RecordingIO(_VERIFY_RESPONSE)
    action = VerifyLeafAction(settings=_settings(**overrides))
    result = asyncio.run(action.execute(g, node_id, io))
    return io, result


# ---------------------------------------------------------------------------
# Flag OFF is byte-identical -- this comes first on purpose
# ---------------------------------------------------------------------------

def test_merge_flag_off_sends_the_shipped_system_message():
    io, _ = _run_merge()
    system = io._system()
    assert SETTINGS["merge_system_prompt"] in system
    assert system.index("goal_achieved") < system.index("goal_evaluation")


def test_merge_flag_off_is_byte_identical_to_no_flag_key_at_all():
    """A default-OFF flag must not perturb the message merely by existing."""
    io_absent, _ = _run_merge()
    io_explicit, _ = _run_merge(merge_goal_evaluation_first_enabled=False)
    assert io_absent._system() == io_explicit._system()


def test_verify_flag_off_sends_the_shipped_system_message():
    io, _ = _run_verify()
    system = io._system()
    assert system == SETTINGS["verify_system_prompt"]
    assert system.index('"verdict"') < system.index('"reasoning"')


# ---------------------------------------------------------------------------
# The variants are the shipped text with exactly one pair swapped
# ---------------------------------------------------------------------------
#
# The anti-fossil guard. The variants are literal constants rather than runtime
# str.replace, because a replace that stops matching after a prompt edit would silently
# make the flag a no-op. These assertions turn that edit into a loud failure instead.

def test_the_merge_variant_is_the_shipped_text_with_only_the_pair_swapped():
    derived = SETTINGS["merge_system_prompt"].replace(
        "goal_achieved: boolean, goal_evaluation: string",
        "goal_evaluation: string, goal_achieved: boolean",
    )
    assert MergeLeafAction._GOAL_EVAL_FIRST_SYSTEM_PROMPT == derived
    assert derived != SETTINGS["merge_system_prompt"], "variant is identical to shipped"


def test_the_verify_variant_is_the_shipped_text_with_only_reasoning_moved():
    derived = (SETTINGS["verify_system_prompt"]
               .replace('{{"verdict"', '{{"reasoning": "<one sentence>", "verdict"')
               .replace(', "reasoning": "<one sentence>"}}', '}}'))
    assert VerifyLeafAction._REASON_FIRST_SYSTEM_PROMPT == derived
    assert derived != SETTINGS["verify_system_prompt"]


def test_the_followup_variant_is_the_shipped_text_with_only_the_pair_swapped():
    """Derived from the SETTINGS text, never from got_operations' inline default -- that
    default is a fossil that never runs, and a variant built from it would ship a prompt
    the engine has never sent."""
    derived = SETTINGS["got_reexpand_followup_system_prompt"].replace(
        '{"needs_followup": boolean, "reason": string}',
        '{"reason": string, "needs_followup": boolean}',
    )
    assert _FOLLOWUP_REASON_FIRST_SYSTEM_PROMPT == derived
    assert derived != SETTINGS["got_reexpand_followup_system_prompt"]


def test_the_verify_variant_tracks_settings_not_the_source_constant():
    """The two shipped copies differ in typography (ASCII hyphen vs em dash, doubled
    braces). Deriving from the source constant would change three things at once and stop
    the A/B isolating field order."""
    assert VerifyLeafAction._REASON_FIRST_SYSTEM_PROMPT.count("—") == 0
    assert " - never on prior knowledge" in VerifyLeafAction._REASON_FIRST_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Merge ON: BOTH ordering sources must flip together
# ---------------------------------------------------------------------------

_HINT_SENTINEL = "Respond with valid JSON"


def test_merge_flag_on_flips_the_template_and_the_schema_hint_together():
    """Merge states the field order twice in one system message. Flipping only one leaves
    the model reading two contradictory orders, which is worse than either ordering."""
    io, _ = _run_merge(merge_goal_evaluation_first_enabled=True)
    system = io._system()
    assert _HINT_SENTINEL in system, "schema hint missing; this test would be vacuous"
    template, hint = system.split(_HINT_SENTINEL, 1)
    assert template.index("goal_evaluation") < template.index("goal_achieved")
    assert hint.index("goal_evaluation") < hint.index("goal_achieved")


def test_merge_flag_on_leaves_the_planning_addendum_intact():
    io, _ = _run_merge(merge_goal_evaluation_first_enabled=True)
    assert SETTINGS["merge_planning_addendum"] in io._system()


def test_merge_flag_on_changes_nothing_but_the_ordering():
    off, _ = _run_merge()
    on, _ = _run_merge(merge_goal_evaluation_first_enabled=True)
    normalize = lambda s: (s.replace("goal_evaluation: string, goal_achieved: boolean",
                                     "goal_achieved: boolean, goal_evaluation: string")
                            .replace('"goal_evaluation"', "@").replace('"goal_achieved"', "#"))
    assert sorted(normalize(off._system())) == sorted(normalize(on._system()))


def _run_merge_with(settings):
    g, node_id = _merge_graph()
    io = _RecordingIO(_MERGE_RESPONSE)
    result = asyncio.run(MergeLeafAction(settings=settings).execute(g, node_id, io))
    return io, result


def test_merge_flag_on_does_not_resurrect_the_no_prompt_concatenation_path():
    """The flag substitutes only when a base template already resolved. Without the guard
    it would turn an empty template into a non-empty one, changing control flow rather than
    field order -- the fallback below would silently become an LLM call.

    Both prompt keys are cleared here, because the planning addendum alone is enough to
    refill ``system_template`` and keep the fallback out of reach."""
    settings = _settings(merge_goal_evaluation_first_enabled=True)
    settings["merge_system_prompt"] = ""
    settings["merge_planning_addendum"] = ""
    io, result = _run_merge_with(settings)
    assert io.payloads == [], "flag resurrected the no-prompt path into an LLM call"
    assert result.get("child_count") == 2


def test_the_two_ordering_sources_are_independent():
    """With the template emptied but the schema present, the flag still flips the hint.

    That is correct, not a leak: the schema hint states the field order on its own, so it
    remains an ordering instruction the model reads even with no template. The pair is
    'both flip together when both exist', not 'the schema follows the template'."""
    def build(flag):
        s = _settings(merge_goal_evaluation_first_enabled=flag)
        s["merge_system_prompt"] = ""
        return _run_merge_with(s)[0]._system()
    off, on = build(False), build(True)
    assert off != on
    assert off.index("goal_achieved") < off.index("goal_evaluation")
    assert on.index("goal_evaluation") < on.index("goal_achieved")


# ---------------------------------------------------------------------------
# Verify ON
# ---------------------------------------------------------------------------

def test_verify_flag_on_puts_reasoning_before_the_verdict():
    io, _ = _run_verify(verify_reason_first_enabled=True)
    system = io._system()
    assert system.index('"reasoning"') < system.index('"verdict"')
    assert system == VerifyLeafAction._REASON_FIRST_SYSTEM_PROMPT


def test_verify_flag_on_keeps_the_one_sentence_bound():
    """The 1024-token verify budget is spent on prose BEFORE the verdict when this flag is
    on, so dropping the length bound would invite truncation into the UNVERIFIABLE
    JSONDecodeError branch."""
    assert "<one sentence>" in VerifyLeafAction._REASON_FIRST_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Emission order must not reach the parser
# ---------------------------------------------------------------------------

def test_merge_parsing_is_unaffected_by_emission_order():
    reordered = json.dumps({
        "summary": "s", "key_findings": ["k"], "goal_evaluation": "done",
        "goal_achieved": True, "missing_requirements": [],
    })
    g, node_id = _merge_graph()
    io = _RecordingIO(reordered)
    result = asyncio.run(
        MergeLeafAction(settings=_settings(merge_goal_evaluation_first_enabled=True))
        .execute(g, node_id, io))
    assert result.get("goal_achieved") is True
    assert result.get("goal_evaluation") == "done"


def test_verify_parsing_is_unaffected_by_emission_order():
    reordered = json.dumps({
        "reasoning": "r", "verdict": "TRUE", "confidence": 0.9,
        "supporting_url": "https://e.example", "contradicting_url": "", "quote": "q",
    })
    g, node_id = _verify_graph()
    io = _RecordingIO(reordered)
    result = asyncio.run(
        VerifyLeafAction(settings=_settings(verify_reason_first_enabled=True))
        .execute(g, node_id, io))
    assert result.get("verdict") == "TRUE"
    assert result.get("reasoning") == "r"


# ---------------------------------------------------------------------------
# The merge schema variant is a pure permutation
# ---------------------------------------------------------------------------

def test_merge_schema_variant_permutes_properties_and_changes_nothing_else():
    base, variant = MERGE_JSON_SCHEMA["schema"], MERGE_JSON_SCHEMA_GOAL_EVAL_FIRST["schema"]
    assert set(base["properties"]) == set(variant["properties"])
    assert base["properties"] == variant["properties"]           # same sub-schemas
    assert base["required"] == variant["required"]               # goal_evaluation stays optional
    assert base["additionalProperties"] == variant["additionalProperties"]
    assert MERGE_JSON_SCHEMA["name"] == MERGE_JSON_SCHEMA_GOAL_EVAL_FIRST["name"]
    keys = list(variant["properties"])
    assert keys.index("goal_evaluation") < keys.index("goal_achieved")
    assert list(base["properties"]).index("goal_achieved") < list(base["properties"]).index("goal_evaluation")


def test_merge_schema_variant_is_not_registered_as_a_default():
    """Registering it would make the reordered hint ship by default, which is exactly what
    'flagged, default OFF' rules out.

    Identity, not equality: dict ``==`` ignores insertion order, so the variant compares
    EQUAL to the shipped schema. json.dumps' order-preserving output is the only thing
    carrying the field order to the model, which is why the permutation test below asserts
    on ``list(properties)`` rather than on the dicts."""
    assert all(v is not MERGE_JSON_SCHEMA_GOAL_EVAL_FIRST for v in DEFAULT_JSON_SCHEMAS.values())
    assert DEFAULT_JSON_SCHEMAS["merge_json_schema"] is MERGE_JSON_SCHEMA
    assert MERGE_JSON_SCHEMA_GOAL_EVAL_FIRST == MERGE_JSON_SCHEMA, (
        "the variant must be a pure permutation, so dict equality holds by construction"
    )


# ---------------------------------------------------------------------------
# The variants must never acquire a settings copy
# ---------------------------------------------------------------------------

def test_no_variant_text_leaks_into_any_settings_file():
    """Standing guard against re-creating the shadowing trap. If a variant ever gains a
    settings key, ``settings.get`` starts preferring it and the source constant becomes a
    fossil -- the exact bug that made got_operations' inline followup default dead code."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "app"
    variants = (
        MergeLeafAction._GOAL_EVAL_FIRST_SYSTEM_PROMPT,
        VerifyLeafAction._REASON_FIRST_SYSTEM_PROMPT,
        _FOLLOWUP_REASON_FIRST_SYSTEM_PROMPT,
    )
    for path in root.glob("idea_dag_settings*.json"):
        name = path.name
        raw = path.read_text()
        for variant in variants:
            assert variant not in raw, f"{name} gained a copy of a variant prompt"
        assert "reason_first_system_prompt" not in raw
        assert "goal_eval_first_system_prompt" not in raw


# ---------------------------------------------------------------------------
# The flags themselves: default OFF, independent
# ---------------------------------------------------------------------------

def test_reason_first_ordering_flags_default_off():
    # Reason-before-answer field ordering at three call sites (promptbench v2, 2026-08-19).
    # All three opt-in and default OFF, from an empty load AND from the shipped production
    # JSON, so every affected prompt stays byte-identical until a flag is set. The
    # pre-registered gate authorized implementation, not enablement.
    from agent.app.idea_policies.config import GoTConfig, MergeConfig, VerifyConfig
    prod = load_idea_dag_settings()

    assert MergeConfig.from_settings({}).goal_evaluation_first_enabled is False
    assert MergeConfig.from_settings(prod).goal_evaluation_first_enabled is False
    assert MergeConfig.from_settings(
        {"merge_goal_evaluation_first_enabled": 1}
    ).goal_evaluation_first_enabled is True

    assert VerifyConfig.from_settings({}).reason_first_enabled is False
    assert VerifyConfig.from_settings(prod).reason_first_enabled is False
    assert VerifyConfig.from_settings(
        {"verify_reason_first_enabled": "true"}
    ).reason_first_enabled is True

    # GoTConfig derives its JSON key by prefixing got_, so the key is not spelled in _KEYS.
    assert GoTConfig.from_settings({}).reexpand_followup_reason_first_enabled is False
    assert GoTConfig.from_settings(prod).reexpand_followup_reason_first_enabled is False
    assert GoTConfig.from_settings(
        {"got_reexpand_followup_reason_first_enabled": 1}
    ).reexpand_followup_reason_first_enabled is True


def test_reason_first_flags_are_independent_of_each_other():
    # Per-call-site flags exist because promptbench v2 showed the effect does NOT
    # generalise across decision types (A4 helped verify +0.084 and hurt select -0.105).
    # Enabling one must never arm another, or a transfer A/B can only report the bundle.
    from agent.app.idea_policies.config import GoTConfig, MergeConfig, VerifyConfig
    only_merge = {"merge_goal_evaluation_first_enabled": True}
    assert MergeConfig.from_settings(only_merge).goal_evaluation_first_enabled is True
    assert VerifyConfig.from_settings(only_merge).reason_first_enabled is False
    assert GoTConfig.from_settings(only_merge).reexpand_followup_reason_first_enabled is False
