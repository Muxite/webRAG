"""N7: an optional structured slot for the answer value in the finalize schema.

``FINAL_JSON_SCHEMA`` is ``{deliverable, summary}``: a count, an argmax or a per-entity list
has nowhere to land except prose, which is the root enabler of the F3 inversion (the model
puts a label in ``deliverable`` and the content in ``summary``). The alternate schema adds
``answer`` and ``per_entity`` and is selected by ``final_answer_slot_enabled``, mirroring
``MERGE_JSON_SCHEMA_GOAL_EVAL_FIRST``: opt-in, default OFF, so shipped behaviour is
byte-identical until it is A/B'd.

Offline: fake IO, no engine, no network.
"""
from __future__ import annotations

import asyncio
import json

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_schemas import (
    DEFAULT_JSON_SCHEMAS,
    FINAL_JSON_SCHEMA,
    FINAL_JSON_SCHEMA_WITH_ANSWER,
)
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_finalize import build_final_payload
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus
from agent.app.idea_policies.config import IdeaConfig


_MANDATE = "How many dams opened before 1960?"
_URL = "https://en.wikipedia.org/wiki/List_of_dams"
_DELIVERABLE = (
    "Three dams opened before 1960: Grand Coulee (1942), McNary (1954) and Chief Joseph "
    "(1955). The Dalles followed in 1957, bringing the pre-1960 total to four."
)


# --------------------------------------------------------------------------- schema shape


def _strict_violations(schema, path="root"):
    out = []
    if schema.get("type") == "object":
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is not False:
            out.append(f"{path}: additionalProperties is not False")
        missing = set(props) - set(schema.get("required", []))
        if missing:
            out.append(f"{path}: required omits {sorted(missing)}")
        for key, sub in props.items():
            out += _strict_violations(sub, f"{path}.{key}")
    if schema.get("type") == "array" and isinstance(schema.get("items"), dict):
        out += _strict_violations(schema["items"], f"{path}[]")
    return out


def test_the_alternate_schema_adds_the_answer_slots():
    props = FINAL_JSON_SCHEMA_WITH_ANSWER["schema"]["properties"]
    assert set(props) == {"answer", "per_entity", "deliverable", "summary"}


def test_the_answer_slot_comes_before_the_prose_fields():
    """The value is committed first, then written up -- the reason-first pattern."""
    order = list(FINAL_JSON_SCHEMA_WITH_ANSWER["schema"]["properties"])
    assert order.index("answer") < order.index("deliverable")


def test_the_prose_field_descriptions_are_unchanged():
    base = FINAL_JSON_SCHEMA["schema"]["properties"]
    alt = FINAL_JSON_SCHEMA_WITH_ANSWER["schema"]["properties"]
    for name in ("deliverable", "summary"):
        assert alt[name] == base[name]


def test_every_new_field_carries_a_description():
    props = FINAL_JSON_SCHEMA_WITH_ANSWER["schema"]["properties"]
    for name in ("answer", "per_entity"):
        assert props[name].get("description", "").strip(), f"{name} has no description"


def test_the_alternate_schema_is_strict_mode_clean():
    """Finalize sends its schema as a real json_schema response_format, unlike merge."""
    assert _strict_violations(FINAL_JSON_SCHEMA_WITH_ANSWER["schema"]) == []


def test_the_alternate_schema_is_not_registered_as_a_default():
    assert all(v is not FINAL_JSON_SCHEMA_WITH_ANSWER for v in DEFAULT_JSON_SCHEMAS.values())
    assert DEFAULT_JSON_SCHEMAS["final_json_schema"] is FINAL_JSON_SCHEMA


def test_the_default_schema_still_has_only_the_two_prose_fields():
    assert set(FINAL_JSON_SCHEMA["schema"]["properties"]) == {"deliverable", "summary"}


# --------------------------------------------------------------------------- config


def test_the_flag_defaults_off():
    assert IdeaConfig.from_settings({}).final.answer_slot_enabled is False
    assert load_idea_dag_settings()["final_answer_slot_enabled"] is False


def test_the_settings_key_is_wired():
    cfg = IdeaConfig.from_settings({"final_answer_slot_enabled": True})
    assert cfg.final.answer_slot_enabled is True


# --------------------------------------------------------------------------- runtime


class _FakeIO:
    def __init__(self, response):
        self._response = response
        self.json_schema = "unset"

    def build_llm_payload(self, messages=None, json_schema=None, **kw):
        self.json_schema = json_schema
        return {"messages": messages}

    async def query_llm_with_fallback(self, payload, model_name=None, fallback_model=None,
                                      timeout_seconds=None):
        return self._response


def _graph() -> IdeaDag:
    g = IdeaDag(root_title="root")
    g.get_node(g.root_id()).details["mandate"] = _MANDATE
    g.add_child(
        g.root_id(), "visit the list page",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.ACTION_RESULT.value: {
                "success": True, "action": IdeaActionType.VISIT.value,
                "url": _URL, "title": "List of dams", "content": "Grand Coulee 1942.",
            },
        },
        status=IdeaNodeStatus.DONE,
    )
    return g


def _run(body, enabled=None):
    settings = load_idea_dag_settings()
    if enabled is not None:
        settings["final_answer_slot_enabled"] = enabled
    io = _FakeIO(json.dumps(body))
    payload = asyncio.run(build_final_payload(io, settings, _graph(), _MANDATE, "m"))
    return io, payload


_WITH_ANSWER = {
    "answer": "4",
    "per_entity": [{"entity": "Grand Coulee", "value": "1942", "source_url": _URL}],
    "deliverable": _DELIVERABLE,
    "summary": "Opened the list article and read the completion years.",
}
_PLAIN = {"deliverable": _DELIVERABLE, "summary": "Opened the list article."}


def test_flag_off_sends_the_shipped_schema_and_ignores_the_extra_fields():
    io, payload = _run(_WITH_ANSWER)

    assert io.json_schema == FINAL_JSON_SCHEMA
    assert payload["final_deliverable"] == _DELIVERABLE
    assert "structured_answer" not in payload
    assert "per_entity" not in payload


def test_flag_off_payload_is_identical_with_and_without_the_extra_fields():
    _, with_extra = _run(_WITH_ANSWER)
    _, without = _run(_PLAIN)

    assert set(with_extra) == set(without)


def test_flag_on_sends_the_alternate_schema():
    io, _ = _run(_WITH_ANSWER, enabled=True)

    assert io.json_schema == FINAL_JSON_SCHEMA_WITH_ANSWER


def test_flag_on_records_the_structured_answer():
    _, payload = _run(_WITH_ANSWER, enabled=True)

    assert payload["structured_answer"] == "4"
    assert payload["per_entity"] == _WITH_ANSWER["per_entity"]
    assert payload["final_deliverable"] == _DELIVERABLE


def test_flag_on_tolerates_a_response_without_the_new_fields():
    _, payload = _run(_PLAIN, enabled=True)

    assert payload["final_deliverable"] == _DELIVERABLE
    assert "structured_answer" not in payload
    assert "per_entity" not in payload


def test_flag_on_ignores_empty_new_fields():
    _, payload = _run({**_PLAIN, "answer": "  ", "per_entity": []}, enabled=True)

    assert "structured_answer" not in payload
    assert "per_entity" not in payload


def test_flag_on_tolerates_a_wrongly_typed_answer():
    _, payload = _run({**_PLAIN, "answer": 4, "per_entity": "not a list"}, enabled=True)

    assert payload["structured_answer"] == "4"
    assert "per_entity" not in payload
