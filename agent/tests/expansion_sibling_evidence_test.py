"""Offline tests for the sibling EVIDENCE digest in the expansion prompt — no LLM.

``run_policy_sibling_evidence_digest_enabled`` is Phase 0's `graph_shared_context` axis
(docs/DAG_V3_LEDGER_MASTER_PLAN_2026-08-25.md section 3). Expansion context is root-ward only
(``IdeaDag.path_to_root``), so a node being expanded cannot see the searches and visits a SIBLING
branch already executed — the diagnosed cause of task 123's churn. This block renders them.

The load-bearing claims:

* flag off (the shipped default) is BYTE-IDENTICAL even with a fully executed sibling sub-tree
  sitting in the graph;
* it is INDEPENDENT of ``run_policy_sibling_context_delta``/``run_policy_ledger_mode`` — that pair
  renders the ledger ROSTER, and an ablation that cannot separate the two measures two things;
* only NON-ancestor nodes contribute (everything root-ward is already in the context verbatim);
* both caps drop WHOLE entries, so a URL or query is never cut into a different-looking one.
"""
from __future__ import annotations

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.idea_policies.config import IdeaConfig
from agent.app.idea_policies.expansion import (
    _SIBLING_EVIDENCE_MAX_ENTRIES,
    _format_sibling_evidence_entries,
    LlmExpansionPolicy,
)

HEADER = "WHAT OTHER BRANCHES OF THIS RUN ALREADY DID"
ON = {"run_policy_sibling_evidence_digest_enabled": True}


class FakeIO:
    telemetry = None


def _visit_details(url: str, ok: bool = True) -> dict:
    return {
        DetailKey.ACTION.value: IdeaActionType.VISIT.value,
        "url": url,
        DetailKey.ACTION_RESULT.value: {
            "success": ok,
            "url": url,
            "page_title": "Some Page",
            "content_total_chars": 1234,
            "links_count": 7,
        },
    }


def _search_details(query: str) -> dict:
    return {
        DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
        DetailKey.QUERY.value: query,
        DetailKey.ACTION_RESULT.value: {
            "success": True,
            "results": [{"url": "https://example.com/hit"}],
        },
    }


def _graph_with_two_branches() -> tuple:
    """Root -> {branch A (search+visit, executed), branch B (the node being expanded)}."""
    graph = IdeaDag(root_title="Research", root_details={"mandate": "Research"})
    root = graph.root_id()
    a = graph.add_child(root, "Arm A", details=_search_details("candidate_a filing")).node_id
    graph.add_child(a, "Arm A visit", details=_visit_details("https://example.com/a"))
    b = graph.add_child(root, "Arm B", details={"goal": "handle candidate_b"}).node_id
    return graph, b


def _system(graph: IdeaDag, node_id: str, **settings) -> str:
    policy = LlmExpansionPolicy(io=FakeIO(), model_name="m", settings=settings or None)
    messages = policy._build_messages(graph, graph.get_node(node_id))
    return next(m["content"] for m in messages if m["role"] == "system")


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------


def test_the_flag_ships_absent_and_therefore_off():
    settings = load_idea_dag_settings()
    assert "run_policy_sibling_evidence_digest_enabled" not in settings
    assert IdeaConfig.from_settings(settings).run_policy.sibling_evidence_digest_enabled is False


# --------------------------------------------------------------------------------------
# flag off: byte-identical
# --------------------------------------------------------------------------------------


def test_executed_siblings_do_not_leak_into_the_prompt_when_the_flag_is_off():
    graph, node_id = _graph_with_two_branches()
    off = _system(graph, node_id)
    assert HEADER not in off
    # ... and the ledger flags do not switch this one on either: they are separate axes.
    ledgered = _system(
        graph,
        node_id,
        run_policy_sibling_context_delta=True,
        run_policy_ledger_mode="observe",
    )
    assert HEADER not in ledgered


def test_the_digest_needs_no_ledger():
    """The whole point of the separate flag: context visibility without ledger-observe mode."""
    graph, node_id = _graph_with_two_branches()
    system = _system(graph, node_id, **ON)
    assert HEADER in system
    assert "[Ledger]" not in system


# --------------------------------------------------------------------------------------
# flag on
# --------------------------------------------------------------------------------------


def test_the_block_names_the_sibling_query_and_url():
    graph, node_id = _graph_with_two_branches()
    system = _system(graph, node_id, **ON)
    assert "candidate_a filing" in system
    assert "https://example.com/a" in system


def test_rootward_nodes_are_excluded_as_already_in_context():
    graph = IdeaDag(root_title="Research", root_details={"mandate": "Research"})
    root = graph.root_id()
    parent = graph.add_child(root, "Parent", details=_search_details("ancestor query")).node_id
    child = graph.add_child(parent, "Child", details={"goal": "next"}).node_id
    system = _system(graph, child, **ON)
    # The only executed node is an ANCESTOR -> nothing to add, prompt unchanged.
    assert HEADER not in system
    assert system == _system(graph, child)


def test_unexecuted_siblings_contribute_nothing():
    graph = IdeaDag(root_title="Research", root_details={"mandate": "Research"})
    root = graph.root_id()
    graph.add_child(root, "Planned but unrun", details={"goal": "later"})
    node_id = graph.add_child(root, "Arm B", details={"goal": "now"}).node_id
    assert HEADER not in _system(graph, node_id, **ON)


def test_duplicate_sibling_actions_are_reported_once():
    graph = IdeaDag(root_title="Research", root_details={"mandate": "Research"})
    root = graph.root_id()
    for _ in range(3):
        graph.add_child(root, "Arm", details=_visit_details("https://example.com/dup"))
    node_id = graph.add_child(root, "Arm B", details={"goal": "now"}).node_id
    system = _system(graph, node_id, **ON)
    # One ENTRY (the rendered outcome repeats the URL inside the same line, which is fine).
    assert system.count("- [visit] https://example.com/dup") == 1


def test_a_failed_sibling_action_is_reported_as_failed():
    graph = IdeaDag(root_title="Research", root_details={"mandate": "Research"})
    root = graph.root_id()
    details = _visit_details("https://example.com/dead", ok=False)
    details[DetailKey.ACTION_RESULT.value]["error"] = "403 forbidden"
    graph.add_child(root, "Arm A", details=details)
    node_id = graph.add_child(root, "Arm B", details={"goal": "now"}).node_id
    system = _system(graph, node_id, **ON)
    assert "FAILED" in system
    assert "https://example.com/dead" in system


# --------------------------------------------------------------------------------------
# bounding
# --------------------------------------------------------------------------------------


def test_entry_cap_drops_whole_oldest_entries():
    lines = [f"- [visit] https://example.com/{i:02d} -> ok" for i in range(_SIBLING_EVIDENCE_MAX_ENTRIES + 4)]
    block = _format_sibling_evidence_entries(lines, budget=100000)
    assert "... [truncated] (4 earlier step(s) omitted)" in block
    for line in lines[-_SIBLING_EVIDENCE_MAX_ENTRIES:]:
        assert line in block
    for line in lines[:4]:
        assert line not in block


def test_character_budget_drops_whole_entries_too():
    lines = [f"- [visit] https://example.com/{i:02d} -> ok" for i in range(4)]
    block = _format_sibling_evidence_entries(lines, budget=len(lines[0]) + 1)
    assert lines[-1] in block
    assert lines[0] not in block
    # Never a partial URL: every surviving line is whole.
    for rendered in block.splitlines():
        assert rendered in lines or rendered.startswith(("WHAT OTHER", "... [truncated]"))


def test_no_entries_renders_nothing():
    assert _format_sibling_evidence_entries([], budget=1000) == ""
    assert _format_sibling_evidence_entries(["   "], budget=1000) == ""


def test_the_budget_is_the_ancestor_content_budget():
    """The digest respects the same per-ancestor character discipline as page content."""
    graph = IdeaDag(root_title="Research", root_details={"mandate": "Research"})
    root = graph.root_id()
    for i in range(_SIBLING_EVIDENCE_MAX_ENTRIES):
        graph.add_child(root, f"Arm {i}", details=_visit_details(f"https://example.com/{i}"))
    node_id = graph.add_child(root, "Arm B", details={"goal": "now"}).node_id
    tight = _system(graph, node_id, expansion_ancestor_content_chars=60, **ON)
    loose = _system(graph, node_id, expansion_ancestor_content_chars=100000, **ON)
    assert loose.count("https://example.com/") > tight.count("https://example.com/")
