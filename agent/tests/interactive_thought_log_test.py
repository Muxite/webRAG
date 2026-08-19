"""Offline unit tests for agent.app.interactive.thought_log.ThoughtLog.

No network, no live LLM: writes to a tmp_path file and inspects the raw text.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "services"))

from agent.app.interactive.thought_log import ThoughtLog, from_env
from agent.app.interactive.thoughts import Thought


def _thought(step=0, node_id="n0"):
    return Thought(
        step_index=step,
        node_id=node_id,
        decisions=[{"stage": "expansion", "chosen": "child-1", "rationale": "best score"}],
        events=[],
        llm_calls=[{"model": "fake-model", "prompt_text": "hi", "completion_text": "hello"}],
        truncated=False,
    )


def test_header_written_once(tmp_path):
    path = tmp_path / "thoughts.log"
    log = ThoughtLog(str(path), label="mandate: do the thing")

    log.write(_thought(step=0))
    log.write(_thought(step=1))

    content = path.read_text(encoding="utf-8")
    assert content.count("agent-debug thought log") == 1
    assert "mandate: do the thing" in content


def test_append_across_multiple_writes(tmp_path):
    path = tmp_path / "thoughts.log"
    log = ThoughtLog(str(path))

    log.write(_thought(step=0, node_id="n0"))
    log.write(_thought(step=1, node_id="n1"))
    log.write(_thought(step=2, node_id="n2"))

    content = path.read_text(encoding="utf-8")
    assert "Step 0" in content
    assert "Step 1" in content
    assert "Step 2" in content


def test_no_ansi_escapes(tmp_path):
    path = tmp_path / "thoughts.log"
    log = ThoughtLog(str(path))
    log.write(_thought())

    content = path.read_text(encoding="utf-8")
    assert "\x1b" not in content


def test_from_env_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("IDEA_THOUGHT_LOG", raising=False)
    assert from_env() is None


def test_from_env_returns_log_when_set(tmp_path, monkeypatch):
    path = tmp_path / "envlog.log"
    monkeypatch.setenv("IDEA_THOUGHT_LOG", str(path))

    log = from_env(label="run-123")
    assert isinstance(log, ThoughtLog)

    log.write(_thought())
    content = path.read_text(encoding="utf-8")
    assert "run-123" in content


def test_classmethod_from_env_alias(tmp_path, monkeypatch):
    path = tmp_path / "envlog2.log"
    monkeypatch.setenv("IDEA_THOUGHT_LOG", str(path))
    log = ThoughtLog.from_env()
    assert isinstance(log, ThoughtLog)


def test_io_failure_does_not_raise():
    # Directory path used as a "file" path: open() will fail on write.
    log = ThoughtLog("/nonexistent-dir-xyz/thoughts.log")
    log.write(_thought())  # must not raise
    log.write(_thought())  # second call: disabled, still must not raise
