import pytest
from app.tick_output import TickOutput

def test_basic():
    o = TickOutput({
        "history_update": "a",
        "note_update": "b",
        "cache_update": {"x": "y", "z": "w"},
        "next_action": " go ",
        "data": "fish, car, bread"
    })
    assert o.show_next_action() == "go"
    assert o.show_requested_data_topics() == ["fish", "car", "bread"]
    assert len(o.to_vector_records()) == 2


def test_empty():
    o = TickOutput({})
    assert o.show_next_action() is None
    assert o.show_requested_data_topics() == []
    assert o.to_vector_records() == []


def test_whitespace_and_summary():
    o = TickOutput({
        "next_action": " run ",
        "data": " a ,  b , , c ",
        "cache_update": {"good": "ok", "bad": " "}
    })
    assert o.show_next_action() == "run"
    assert o.show_requested_data_topics() == ["a", "b", "c"]
    v = o.to_vector_records()
    assert v == [{"tag": "good", "content": "ok"}]