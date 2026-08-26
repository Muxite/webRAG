"""Adversarial fixture for the finalize field-swap failure (deliverable vs summary).

Observed live (21 cells, qwen2.5:7b, ``good_adaptive``): 3 of 21 rep-1 cells wrote a bare
title fragment into ``deliverable`` (``"Skiathos Airport runway length"``, ``"Edinburgh"``)
while the full, correct answer went into ``summary`` — which nothing downstream reads. The
run looked like a total extraction failure even though the evidence gathered was fine.

The cause is prompt/schema ambiguity: ``FINAL_JSON_SCHEMA`` carried no field descriptions
and the finalize system prompt only listed the two key names. So the contract asserted here
is a *text* contract — the schema and the prompt must each state, unambiguously, which field
holds the answer. There is no runtime gate to test: a swap is indistinguishable downstream
from a genuinely terse deliverable, so the fix has to live in what the model is told.

The fixture below pins the shape of the failure (and that it still ships as-is), so anyone
who later adds a runtime guard has the exact case to work against.
"""
from __future__ import annotations

import asyncio
import json

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_schemas import FINAL_JSON_SCHEMA
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_finalize import build_final_payload
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus
from agent.app.prompt_builder import FinalPromptBuilder


_MANDATE = "Search for the runway length of Skiathos Airport and visit the page. How long is it?"
_URL = "https://en.wikipedia.org/wiki/Skiathos_International_Airport"
# The real swap: label in `deliverable`, the answer stranded in `summary`.
_SWAPPED_DELIVERABLE = "Skiathos Airport runway length"
_SWAPPED_SUMMARY = (
    "The runway at Skiathos International Airport is 1,628 metres long, per the airport's "
    "Wikipedia page."
)


class _FakeIO:
    def __init__(self, response):
        self._response = response

    def build_llm_payload(self, messages=None, **kw):
        return {"messages": messages}

    async def query_llm_with_fallback(self, payload, model_name=None, fallback_model=None,
                                      timeout_seconds=None):
        return self._response


def _grounded_graph() -> IdeaDag:
    g = IdeaDag(root_title="root")
    g.get_node(g.root_id()).details["mandate"] = _MANDATE
    g.add_child(
        g.root_id(), "visit the airport page",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.ACTION_RESULT.value: {
                "success": True, "action": IdeaActionType.VISIT.value,
                "url": _URL, "title": "Skiathos International Airport",
                "content": "The runway is 1,628 m long.",
            },
        },
        status=IdeaNodeStatus.DONE,
    )
    return g


def _run(deliverable, summary):
    response = json.dumps({"deliverable": deliverable, "summary": summary})
    settings = load_idea_dag_settings()
    return asyncio.run(
        build_final_payload(_FakeIO(response), settings, _grounded_graph(), _MANDATE, "m")
    )


def _fields():
    return FINAL_JSON_SCHEMA["schema"]["properties"]


def test_both_final_fields_carry_a_non_empty_description():
    for name, field in _fields().items():
        description = field.get("description", "")
        assert description.strip(), f"final schema field {name!r} has no description"


def test_the_deliverable_description_forbids_a_bare_label():
    description = _fields()["deliverable"]["description"].lower()
    # It must say what the field IS...
    assert "complete final answer" in description
    # ...and rule out the exact shapes the 7B model substituted.
    for excluded in ("not a title", "label", "one-line summary"):
        assert excluded in description


def test_the_summary_description_forbids_the_answer():
    description = _fields()["summary"]["description"].lower()
    assert "what actions were taken" in description
    assert "not the answer itself" in description


def test_the_finalize_prompt_states_the_distinction_inline():
    """Schema metadata alone is weak leverage on a local model, so the prompt must repeat it."""
    instructions = FinalPromptBuilder.SYSTEM_INSTRUCTIONS.lower()
    assert "'deliverable'" in instructions and "'summary'" in instructions
    assert "complete answer itself" in instructions
    assert "never put a bare title" in instructions
    assert "never the answer content itself" in instructions


def test_a_swapped_response_ships_the_label_and_strands_the_answer():
    """Pins the failure: nothing downstream reads `summary`, so the answer is simply lost."""
    payload = _run(_SWAPPED_DELIVERABLE, _SWAPPED_SUMMARY)
    assert payload["final_deliverable"] == _SWAPPED_DELIVERABLE
    assert "1,628" not in payload["final_deliverable"]
    assert payload["action_summary"] == _SWAPPED_SUMMARY


def test_a_correctly_filled_response_carries_the_answer():
    payload = _run(_SWAPPED_SUMMARY, "Searched for and opened the airport's Wikipedia page.")
    assert "1,628" in payload["final_deliverable"]
