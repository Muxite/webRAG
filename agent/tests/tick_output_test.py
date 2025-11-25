import pytest
from app.tick_output import TickOutput, ActionType
from typing import List


def test_basic():
    o = TickOutput({
        "history_update": "a",
        "note_update": "b",
        "next_action": " search, best cars",
        "cache_retrieved": ["fish", "car", "bread"],
        "cache_update": [{"document": "ok", "metadata": {"tag": "good"}},
                         {"document": "1", "metadata": {"1": "1"}},
                         {"document": "2", "metadata": {"3": "4"}}
                         ],
        "deliverable": "https://www.link.com/"
    })
    assert o.show_next_action() == (ActionType.SEARCH, "best cars")
    assert o.show_requested_data_topics() == ["fish", "car", "bread"]
    assert o.deliverable() == "https://www.link.com/"

    assert o.get_vector_documents() == ["ok", "1", "2"]
    assert o.get_vector_metadatas() == [{"tag": "good"}, {"1": "1"}, {"3": "4"}]
    assert isinstance(o.get_vector_ids(), List)
    assert len(o.get_vector_ids()) == 3

def test_empty():
    o = TickOutput({})
    assert o.show_next_action() == (ActionType.THINK, None)
    assert o.show_requested_data_topics() == []
    assert o.get_vector_ids() == []


def test_default():
    o = TickOutput({
        "next_action": " eggs ",
        "cache_retrieved": ["a", "b", "c"],
    })
    assert o.show_next_action() == (ActionType.THINK, None)
    assert o.show_requested_data_topics() == ["a", "b", "c"]
