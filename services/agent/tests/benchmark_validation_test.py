from agent.app.benchmark_cli import validate_benchmark
from agent.app.telemetry import TelemetrySession


def test_validate_benchmark_requires_activity_and_citations():
    telemetry = TelemetrySession(enabled=True, mandate="test", correlation_id="x")
    output = {"final_deliverable": "No citations here", "action_summary": ""}
    result = validate_benchmark(output, telemetry)
    assert result["passed"] is False
    assert "no_search_activity" in result["reasons"]
    assert "no_visit_activity" in result["reasons"]
    assert "insufficient_citations" in result["reasons"]


def test_validate_benchmark_passes_with_activity():
    telemetry = TelemetrySession(enabled=True, mandate="test", correlation_id="x")
    telemetry.documents_seen.append({"source": "search", "document": {"title": "a"}})
    telemetry.documents_seen.append({"source": "visit", "document": {"content": "x"}})
    output = {
        "final_deliverable": "See https://a.example ref",
        "action_summary": "Also https://b.example and https://c.example",
    }
    result = validate_benchmark(output, telemetry)
    assert result["passed"] is True
