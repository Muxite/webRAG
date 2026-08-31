"""
Offline unit tests for the graph wiki-race shortest-chain task (test 047) — free, no LLM.

Unlike the plain visit-count check used on other tasks, test 047's keystone is grounded by a
structural mechanism: adjacency is verified against the agent's own recorded visit data, not
its self-reported claim. A hallucinated/self-reported chain of real Wikipedia URLs (even if
every URL and adjacency happens to be correct from parametric memory) verifies ZERO hops
unless the corresponding pages were actually visited and their outgoing links captured. So an
"ungrounded but correct-looking" answer already collapses the keystone (and everything gated
on it) to 0, independent of the ``observability.visit.count`` field.

ARM-SYMMETRY FIX (2026-08-31): the adjacency source used to be ``build_visit_link_graph``
alone, which reads exclusively from ``result["graph"]["nodes"]`` -- populated ONLY by the
``graph``/``naive_discretion`` arms (``sequential_react``, ``graph_compiled`` and
``langgraph_react`` always return the empty graph per ``idea_test_utils.visited_evidence``'s
docstring). That meant the keystone was a STRUCTURAL 0 for those three arms regardless of what
they actually visited -- an arm-comparison artifact, not a capability difference. ``_verify``
now sources adjacency from ``idea_test_utils.visit_adjacency_map``, which additionally reads
``observability['evidence']['visited']`` (populated identically by every arm), so a non-graph
arm that genuinely followed the hyperlinks can still earn the keystone. The tests below cover
both the pre-existing graph-arm path and the new non-graph-arm path.
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


def test_non_graph_arm_can_earn_the_keystone_via_observability_evidence():
    """A sequential_react/langgraph_react-style run never populates result['graph'] -- it stays
    the empty placeholder -- but genuinely visited every page in the chain and its fetched text
    contains the next hop's raw URL. Before the arm-symmetry fix this scored 0 no matter what;
    it must now verify, since three of the four execution arms can never populate result['graph']."""
    r = {
        "output": {"final_deliverable": _CHAIN_TEXT},
        "graph": {"nodes": {}},
    }
    observability = {
        "visit": {"count": 3},
        "evidence": {"visited": [
            {"url": "https://en.wikipedia.org/wiki/Pizza",
             "content": "Pizza originates in Italy. See https://en.wikipedia.org/wiki/Italy for the country."},
            {"url": "https://en.wikipedia.org/wiki/Italy",
             "content": "Italy was central to the https://en.wikipedia.org/wiki/Roman_Empire for centuries."},
            {"url": "https://en.wikipedia.org/wiki/Roman_Empire",
             "content": "The Roman Empire was a period of ancient Rome."},
        ]},
    }
    assert t.validate_keystone_chain(r, observability)["score"] == 1.0
    assert t.validate_chain_progress(r, observability)["score"] == 1.0
    assert t.validate_efficiency(r, observability)["score"] == 1.0


def test_non_graph_arm_with_no_evidence_still_gates_to_zero():
    """The arm-symmetry fix only recovers adjacency present in genuinely fetched content -- it
    must not manufacture credit for an arm that claims visits (visit.count > 0) but supplies no
    evidence content at all."""
    r = {"output": {"final_deliverable": _CHAIN_TEXT}, "graph": {"nodes": {}}}
    observability = {"visit": {"count": 5}, "evidence": {"visited": []}}
    assert t.validate_keystone_chain(r, observability)["score"] == 0.0
