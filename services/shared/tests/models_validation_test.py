"""
Unit tests for shared.models validation constraints added in M1.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.models import TaskRequest


def test_task_request_accepts_valid_request():
    req = TaskRequest(mandate="hello", max_ticks=50)
    assert req.mandate == "hello"
    assert req.max_ticks == 50


def test_task_request_defaults_max_ticks():
    req = TaskRequest(mandate="hello")
    assert req.max_ticks == 50


def test_task_request_rejects_empty_mandate():
    with pytest.raises(ValidationError):
        TaskRequest(mandate="", max_ticks=50)


def test_task_request_rejects_mandate_over_8000_chars():
    with pytest.raises(ValidationError):
        TaskRequest(mandate="a" * 8001, max_ticks=50)


def test_task_request_accepts_max_8000_chars():
    req = TaskRequest(mandate="a" * 8000)
    assert len(req.mandate) == 8000


def test_task_request_rejects_max_ticks_below_one():
    with pytest.raises(ValidationError):
        TaskRequest(mandate="hi", max_ticks=0)


def test_task_request_rejects_max_ticks_above_200():
    with pytest.raises(ValidationError):
        TaskRequest(mandate="hi", max_ticks=99999)


def test_task_request_accepts_max_ticks_at_upper_bound():
    req = TaskRequest(mandate="hi", max_ticks=200)
    assert req.max_ticks == 200


def test_task_request_optional_correlation_id():
    req = TaskRequest(mandate="hi", correlation_id="abc-123")
    assert req.correlation_id == "abc-123"
    req_no_id = TaskRequest(mandate="hi")
    assert req_no_id.correlation_id is None
