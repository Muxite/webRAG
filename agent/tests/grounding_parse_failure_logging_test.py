"""D8 (ENGINE_DESIGN_REVIEW): a grounding-gate parse failure must not be silent.

``requires_grounded_answer`` fails OPEN when the mandate parser throws — grounding is then
treated as NOT required, the permissive direction for a safety gate. That default stays
(refusing a whole run over a parser bug is worse), but it used to be logged nowhere at all,
so a real parse bug in this gate was invisible in production.

Pinned here: the exception path logs a WARNING carrying the exception context, and the
return value is unchanged (visibility addition, not a behavior change).
"""
from __future__ import annotations

import agent.app.idea_policies.grounding as grounding
from agent.app.idea_policies.grounding import requires_grounded_answer


_RESEARCH_MANDATE = (
    "Visit https://example.org/lake and report its maximum depth. Do not guess."
)


class _Boom(RuntimeError):
    pass


def _explode(_mandate):
    raise _Boom("mandate regex blew up")


def test_parse_failure_logs_a_warning_with_context(monkeypatch, caplog):
    monkeypatch.setattr(grounding, "parse_mandate_requirements", _explode)
    with caplog.at_level("WARNING"):
        result = requires_grounded_answer(_RESEARCH_MANDATE)
    assert result is False  # fail-open behavior deliberately unchanged
    messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("[GROUNDING]" in m for m in messages)
    hit = next(m for m in messages if "[GROUNDING]" in m)
    assert "_Boom" in hit and "mandate regex blew up" in hit
    assert "maximum depth" in hit  # the mandate under parse is quoted for diagnosis


def test_long_mandate_is_truncated_in_the_log(monkeypatch, caplog):
    monkeypatch.setattr(grounding, "parse_mandate_requirements", _explode)
    with caplog.at_level("WARNING"):
        requires_grounded_answer("x" * 5000)
    hit = next(r.message for r in caplog.records if "[GROUNDING]" in r.message)
    assert len(hit) < 500


def test_healthy_path_logs_nothing(caplog):
    with caplog.at_level("WARNING"):
        assert requires_grounded_answer(_RESEARCH_MANDATE) is True
        assert requires_grounded_answer("Summarize the text I pasted above.") is False
    assert not [r for r in caplog.records if "[GROUNDING]" in r.message]
