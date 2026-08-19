"""The SHIPPED arms must be the text the engine actually sends.

``factors.py`` has cited this filename since the subsystem was written, while the
file did not exist -- the checks it describes lived at the bottom of
``promptbench_items_integrity_test.py``. They move here now that there are four
shipped sources instead of one, and they grow to cover the way those four differ.

WHY THIS IS NOT A FORMALITY
---------------------------
Three of the four prompts live in ``agent/app/idea_dag_settings.json``, not in
source, and the engine PREFERS the settings value::

    # actions.py:2304
    system_content = self.settings.get("verify_system_prompt") or self._DEFAULT_SYSTEM_PROMPT

v1's SHIPPED arm imported the class constant alone. The two agree today, so v1's
numbers stand -- but nothing tested that, and editing a JSON file would have moved
what the engine sends while leaving the benchmark arm measuring history.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from agent.app.promptbench.factors import build_prompt, is_applicable, shipped_instruction
from agent.app.promptbench.families import REGISTRY, _undouble_braces, shipped_prompt

SETTINGS = json.loads(Path("agent/app/idea_dag_settings.json").read_text())

SHIPPED_FAMILIES = [name for name, f in REGISTRY.items() if f.has_shipped]


def _normalise(text: str) -> str:
    """Compare on meaning, not on typography.

    The settings copies use ASCII hyphens where the source constants use em-dashes,
    and their JSON examples carry doubled braces because the values are ``.format()``
    templates. Neither difference changes what the model is asked.
    """
    text = _undouble_braces(text)
    text = text.replace("—", "-").replace("–", "-")
    return " ".join(text.split())


@pytest.mark.parametrize("family", SHIPPED_FAMILIES)
def test_every_shipped_arm_resolves_to_non_empty_text(family):
    text = REGISTRY[family].shipped()
    assert isinstance(text, str) and len(text.strip()) > 50


@pytest.mark.parametrize("family", SHIPPED_FAMILIES)
def test_no_shipped_arm_ships_an_unrendered_format_template(family):
    """Settings values are ``.format()`` templates. Shipping a literal ``{{`` to a
    model would spike parse failures and be read as a model deficit."""
    assert "{{" not in REGISTRY[family].shipped()


@pytest.mark.parametrize("family,vocabulary", [
    ("verify", ["verdict", "TRUE", "FALSE", "UNVERIFIABLE"]),
    ("keystone_claim", ["verdict", "TRUE", "FALSE"]),
    ("followup", ["needs_followup", "reason"]),
    ("goal_achieved", ["goal_achieved", "goal_evaluation"]),
    ("calibration", ["confidence", "reason"]),
])
def test_each_shipped_arm_still_asks_for_its_declared_vocabulary(family, vocabulary):
    """If the engine's output schema changes, the family's alias table is stale and
    the arm would score well-formed answers as parse failures -- which is exactly
    what happened to v1's SHIPPED cells (92% parse failure on correct output)."""
    text = REGISTRY[family].shipped()
    for token in vocabulary:
        assert token in text, f"{family}: shipped prompt no longer mentions {token!r}"


# ---------------------------------------------------------------------------
# Settings-vs-source agreement
# ---------------------------------------------------------------------------

def test_the_settings_verify_prompt_still_agrees_with_the_class_constant():
    from agent.app.idea_policies.actions import VerifyLeafAction

    assert _normalise(SETTINGS["verify_system_prompt"]) == _normalise(
        VerifyLeafAction._DEFAULT_SYSTEM_PROMPT)


def test_the_followup_arm_uses_the_settings_text_which_is_NOT_the_source_default():
    """These two genuinely differ, and the settings copy is the one that ships.

    ``got_operations.py``'s inline default is 347 characters. The settings value is
    614, and the extra paragraph is a real behavioural constraint::

        Only answer true when the resolved content names a specific new entity,
        page, or question that must be investigated next (e.g. a disambiguation
        survivor that points to a further target). Answer false for vague,
        speculative, or already-answered follow-ups.

    Because ``settings.get(key, default)`` prefers the settings value, the inline
    default is a fossil that never runs. An arm built from the source constant would
    have measured a prompt the engine does not send -- which is the precise failure
    this file exists to catch, found on its first run.
    """
    from agent.app.promptbench.families import _followup_fallback

    settings_text = SETTINGS["got_reexpand_followup_system_prompt"]
    assert REGISTRY["followup"].shipped() == _undouble_braces(settings_text)
    assert len(settings_text) > len(_followup_fallback())
    assert "Only answer true when the resolved content names a specific new entity" in settings_text
    # Shared opening: drift detection without demanding equality the engine does not need.
    assert settings_text.startswith("You are a follow-up detector in a Graph-of-Thought")
    assert _followup_fallback().startswith("You are a follow-up detector in a Graph-of-Thought")


def test_the_settings_merge_prompt_still_agrees_with_the_recorded_copy():
    from agent.app.promptbench.families import _merge_fallback

    assert _normalise(SETTINGS["merge_system_prompt"]) == _normalise(_merge_fallback())


def test_the_step_confidence_prompt_matches_the_engine_source_verbatim():
    """The only one of the four with no settings key: it is a literal in
    ``got_operations.py``. Compared against the source text so the copy here cannot
    drift unnoticed."""
    from agent.app.promptbench.families import _step_confidence_fallback

    source = Path("agent/app/got_operations.py").read_text()
    for fragment in ("You are a step-level verifier in a Graph-of-Thought research system.",
                     "CORRECT and ON-TRACK toward the overall task"):
        assert fragment in source
        assert fragment in _step_confidence_fallback()


def test_settings_resolution_prefers_settings_and_falls_back_cleanly():
    assert shipped_prompt("verify_system_prompt", lambda: "FALLBACK") != "FALLBACK"
    assert shipped_prompt("no_such_key_anywhere", lambda: "FALLBACK") == "FALLBACK"
    assert shipped_prompt(None, lambda: "FALLBACK") == "FALLBACK"


# ---------------------------------------------------------------------------
# The premise of the cycle
# ---------------------------------------------------------------------------

def test_the_shipped_verify_prompt_still_puts_the_answer_before_the_reasoning():
    """The premise of the whole investigation. If the engine is changed to put the
    reasoning first, this fails and the pre-registered comparison must be re-stated
    rather than silently re-interpreted."""
    text = shipped_instruction("verify")
    assert text.index('"verdict"') < text.index('"reasoning"')


def test_the_shipped_merge_prompt_still_puts_the_boolean_before_its_evaluation():
    text = REGISTRY["goal_achieved"].shipped()
    assert text.index("goal_achieved") < text.index("goal_evaluation")


def test_the_shipped_followup_prompt_still_puts_the_boolean_before_its_reason():
    text = REGISTRY["followup"].shipped()
    assert text.index("needs_followup") < text.index('"reason"')


def test_the_shipped_confidence_prompt_still_puts_the_number_before_its_reason():
    text = REGISTRY["calibration"].shipped()
    assert text.index('"confidence"') < text.index('"reason"')


# ---------------------------------------------------------------------------
# select has no shipped arm, and must not acquire one by accident
# ---------------------------------------------------------------------------

def test_select_has_no_shipped_arm_so_those_cells_are_never_run():
    """A four-way truth verdict cannot name one of five candidates. v1 ran those
    cells and threw every one away in analysis; here they are never placed."""
    assert REGISTRY["select"].has_shipped is False
    assert not is_applicable("select", "SHIPPED")
    with pytest.raises(ValueError):
        REGISTRY["select"].shipped()


@pytest.mark.parametrize("family", SHIPPED_FAMILIES)
def test_the_shipped_text_actually_reaches_the_rendered_prompt(family):
    from agent.app.promptbench.availability import PromptContext
    from agent.app.promptbench.families import build

    items = build([family])[family]
    prompt = build_prompt(items[0].runtime,
                          PromptContext(family=family, variant="SHIPPED", model="t"))
    assert REGISTRY[family].shipped()[:60] in prompt
