"""
Offline unit tests for the graph wiki-race shortest-chain task (test 047) — free, no LLM.

FLAGGED (no grounding-gate code change applied): unlike the plain visit-count check used on
the other 9 tasks in this batch, test 047's keystone is ALREADY grounded by a strictly
stronger, structural mechanism — ``build_visit_link_graph`` reconstructs the adjacency graph
exclusively from ``action_result`` entries with ``action == "visit"`` and ``success`` in the
agent's own execution graph (``result["graph"]["nodes"]``), not from ``observability`` at all.
A hallucinated/self-reported chain of real Wikipedia URLs (even if every URL and adjacency
happens to be correct from parametric memory) verifies ZERO hops unless the corresponding
pages were actually visited and their outgoing links captured. So an "ungrounded but
correct-looking" answer already collapses the keystone (and everything gated on it) to 0,
independent of the ``observability.visit.count`` field the other 9 tasks now gate on. These
tests demonstrate that inherent immunity directly, plus the grounded-correct path.
"""
from agent.app.idea_tests import test_047_graph_wikirace as t


def _r_grounded(chain_text, edges):
    """Build a result with a real ``graph`` of visited pages and their outgoing links, so the
    reported chain is objectively verifiable."""
    nodes = {}
    for i, (src, outs) in enumerate(edges.items()):
        nodes[str(i)] = {
            "details": {
                "action_result": {
                    "action": "visit",
                    "success": True,
                    "url": src,
                    "urls_visited": [src],
                    "links_full": outs,
                }
            }
        }
    return {"output": {"final_deliverable": chain_text}, "graph": {"nodes": nodes}}


_CHAIN_TEXT = (
    "https://en.wikipedia.org/wiki/Pizza\n"
    "https://en.wikipedia.org/wiki/Italy\n"
    "https://en.wikipedia.org/wiki/Roman_Empire\n"
)

_EDGES = {
    "https://en.wikipedia.org/wiki/Pizza": ["https://en.wikipedia.org/wiki/Italy"],
    "https://en.wikipedia.org/wiki/Italy": ["https://en.wikipedia.org/wiki/Roman_Empire"],
}


def test_grounded_verified_chain_scores_all():
    r = _r_grounded(_CHAIN_TEXT, _EDGES)
    assert t.validate_keystone_chain(r, {"visit": {"count": 2}})["score"] == 1.0
    assert t.validate_chain_progress(r, {"visit": {"count": 2}})["score"] == 1.0
    assert t.validate_efficiency(r, {"visit": {"count": 2}})["score"] == 1.0


def test_hallucinated_correct_looking_chain_with_no_real_visits_gates_to_zero():
    """The 'ungrounded but correct-looking' analogue for this task: the model reports the
    right URLs and adjacency from memory, but the execution graph shows NO visit actions at
    all (a fabricated chain). This must score 0 regardless of any observability.visit.count
    value, because the verification is derived solely from the graph's recorded visits."""
    r = {"output": {"final_deliverable": _CHAIN_TEXT}, "graph": {"nodes": {}}}
    # Even if observability *claims* visits happened, the chain is unverifiable from the graph.
    fake_grounded_obs = {"visit": {"count": 5}}
    assert t.validate_keystone_chain(r, fake_grounded_obs)["score"] == 0.0
    assert t.validate_keystone_chain(r, fake_grounded_obs)["passed"] is False
    assert t.validate_chain_progress(r, fake_grounded_obs)["score"] == 0.0
    assert t.validate_efficiency(r, fake_grounded_obs)["score"] == 0.0


def test_partially_visited_chain_gates_to_zero_but_reports_progress():
    # Only the first hop's page was actually visited; the second hop is unverifiable.
    r = _r_grounded(_CHAIN_TEXT, {"https://en.wikipedia.org/wiki/Pizza": ["https://en.wikipedia.org/wiki/Italy"]})
    result = t.validate_keystone_chain(r, {"visit": {"count": 1}})
    assert result["score"] == 0.0
    assert result["passed"] is False
