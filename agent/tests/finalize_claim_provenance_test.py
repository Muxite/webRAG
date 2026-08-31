"""The VERIFY action computes per-claim evidence -- verdict, supporting URL, a quote -- and
``build_final_payload`` used to throw all of it away: ``output.sources[]`` is a visited-URL log
(``{url, title}``), not provenance for any particular claim.

``claim_provenance`` now carries it into the final payload, with the one check that needs no
model judgment: is the quote LITERALLY in the page we opened. Uncheckable quotes report
``quote_verified: None`` -- never ``True``.

Pinned here as well: ``sources[]`` keeps its exact ``{url, title}`` shape (178 task validators
and the analysis scripts read it).

No network: hand-built graphs plus a scripted finalize response.
"""
from __future__ import annotations

import asyncio
import json

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_finalize import build_final_payload
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus


_MANDATE = "Find the maximum depth of Quesnel Lake and cite the page you read."
_URL = "https://en.wikipedia.org/wiki/Quesnel_Lake"
_PAGE = "Quesnel Lake is in British Columbia.\nIts maximum depth is 511 m,\nmaking it deep."


class _FakeIO:
    def __init__(self, response):
        self._response = response

    def build_llm_payload(self, messages=None, **kw):
        return {"messages": messages}

    async def query_llm_with_fallback(self, payload, model_name=None, fallback_model=None,
                                      timeout_seconds=None):
        return self._response


def _graph(*verify_results, page_url=_URL, page_content=_PAGE) -> IdeaDag:
    g = IdeaDag(root_title="root")
    g.get_node(g.root_id()).details["mandate"] = _MANDATE
    if page_url:
        g.add_child(
            g.root_id(), "visit the lake page",
            details={
                DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                DetailKey.ACTION_RESULT.value: {
                    "success": True, "action": IdeaActionType.VISIT.value,
                    "url": page_url, "title": "Quesnel Lake", "content": page_content,
                },
            },
            status=IdeaNodeStatus.DONE,
        )
    for i, vr in enumerate(verify_results):
        g.add_child(
            g.root_id(), f"verify {i}",
            details={
                DetailKey.ACTION.value: IdeaActionType.VERIFY.value,
                DetailKey.ACTION_RESULT.value: vr,
            },
            status=IdeaNodeStatus.DONE,
        )
    return g


def _verify(claim="Quesnel Lake is 511 m deep", verdict="SUPPORTED", quote="maximum depth is 511 m",
            supporting_url=_URL, success=True):
    return {
        "success": success, "action": IdeaActionType.VERIFY.value,
        "claim": claim, "verdict": verdict, "confidence": 0.9,
        "supporting_url": supporting_url, "contradicting_url": "",
        "quote": quote, "reasoning": "the page says so",
        "evidence_sources": [_URL],
    }


def _run(graph, **overrides):
    settings = load_idea_dag_settings()
    settings.update(overrides)
    response = json.dumps({"deliverable": "511 m.", "summary": "read and verified"})
    return asyncio.run(build_final_payload(_FakeIO(response), settings, graph, _MANDATE, "m"))


# --------------------------------------------------------------- the field exists


def test_verify_output_reaches_the_payload():
    entries = _run(_graph(_verify()))["claim_provenance"]

    assert len(entries) == 1
    assert entries[0] == {
        "claim": "Quesnel Lake is 511 m deep",
        "verdict": "SUPPORTED",
        "supporting_url": _URL,
        "quote": "maximum depth is 511 m",
        "quote_verified": True,
    }


def test_a_run_with_no_verify_actions_reports_an_empty_list():
    assert _run(_graph())["claim_provenance"] == []


def test_failed_verify_nodes_are_not_reported_as_provenance():
    assert _run(_graph(_verify(success=False)))["claim_provenance"] == []


# --------------------------------------------------------------- the quote check


def test_a_quote_present_in_the_page_is_verified():
    entry = _run(_graph(_verify(quote="Quesnel Lake is in British Columbia")))["claim_provenance"][0]

    assert entry["quote_verified"] is True


def test_a_quote_absent_from_the_page_is_marked_false():
    entry = _run(_graph(_verify(quote="its maximum depth is 900 m")))["claim_provenance"][0]

    assert entry["quote_verified"] is False


def test_a_quote_spanning_a_line_break_still_verifies():
    """Extracted page text wraps; the check normalizes whitespace, nothing else."""
    entry = _run(_graph(_verify(quote="British Columbia. Its maximum depth")))["claim_provenance"][0]

    assert entry["quote_verified"] is True


def test_a_quote_whose_page_was_never_opened_is_unknown_not_verified():
    entry = _run(_graph(_verify(supporting_url="https://example.com/other")))["claim_provenance"][0]

    assert entry["quote_verified"] is None


def test_a_verdict_with_no_quote_is_unknown_not_verified():
    entry = _run(_graph(_verify(quote="")))["claim_provenance"][0]

    assert entry["quote"] == ""
    assert entry["quote_verified"] is None


def test_a_verdict_with_no_supporting_url_is_unknown_not_verified():
    entry = _run(_graph(_verify(supporting_url="")))["claim_provenance"][0]

    assert entry["quote_verified"] is None


def test_the_supporting_url_matches_a_visited_page_up_to_fragment_and_case():
    entry = _run(_graph(_verify(supporting_url=_URL.upper() + "#Geography")))["claim_provenance"][0]

    assert entry["quote_verified"] is True


# --------------------------------------------------------------- sources[] is untouched


def test_sources_keeps_its_exact_url_title_shape():
    payload = _run(_graph(_verify()))

    assert payload["sources"] == [{"url": _URL, "title": "Quesnel Lake"}]


def test_a_verify_node_never_becomes_a_source():
    payload = _run(_graph(_verify(), _verify(claim="second claim")))

    assert len(payload["sources"]) == 1
