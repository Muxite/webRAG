"""Deterministic candidate-coverage gate — offline, no LLM.

extract_named_candidates must extract the NAMES from an enumerated candidate list
(test_095-style) and fail OPEN on everything else, crucially on numbered INSTRUCTION
steps (test_051 / test_065) which must NOT be mistaken for candidates.
"""
from __future__ import annotations

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.idea_policies.candidate_coverage import (
    CandidateCoverageResult,
    evaluate_candidate_coverage,
    extract_named_candidates,
    strip_enumerated_items,
)

# The real test_095 STAGE-1 candidate list (numbered "N. Name — description").
AVON_MANDATE = (
    "STAGE 1 — eliminate to one survivor. Britain has four principal rivers named 'Avon':\n"
    "  1. River Avon, Bristol — the Bristol Avon (flows through Bath and Bristol)\n"
    "  2. River Avon, Warwickshire — the Warwickshire or 'Shakespeare's' Avon (flows through Stratford-upon-Avon)\n"
    "  3. River Avon, Hampshire — the Hampshire or 'Salisbury' Avon (flows through Salisbury)\n"
    "  4. River Avon, Strathspey — the Scottish (Strathspey / Banffshire) Avon (rises in the Cairngorms, in Moray)\n"
    "Exactly ONE of these four empties into the ENGLISH CHANNEL."
)

# test_051 numbered INSTRUCTION steps — imperative verbs, not candidates.
INSTRUCTION_MANDATE_051 = (
    "You are given NO URLs. Follow a research chain:\n"
    "  1. Identify the AUTHOR of the novel 'Things Fall Apart'.\n"
    "  2. Read that author's page; identify the UNIVERSITY they attended as an undergraduate.\n"
    "  3. Read that university's page; identify the YEAR it was founded/established.\n"
)

# test_065-style numbered INSTRUCTION steps.
INSTRUCTION_MANDATE_065 = (
    "Follow a dependency chain:\n"
    "  1. Identify the POET who wrote the 1924 collection 'Twenty Love Poems'.\n"
    "  2. Open that poet's page and read their BIRTHPLACE.\n"
    "  3. Open that town's page and read its ELEVATION above sea level.\n"
)


def test_extract_real_candidate_list_names_only():
    names = extract_named_candidates(AVON_MANDATE)
    assert names == [
        "River Avon, Bristol",
        "River Avon, Warwickshire",
        "River Avon, Hampshire",
        "River Avon, Strathspey",
    ]
    # Description text after the em-dash must NOT leak into any name.
    for n in names:
        assert "flows" not in n.lower()
        assert "—" not in n
        assert "(" not in n


def test_plain_prose_no_numbered_list_returns_empty():
    prose = (
        "Determine which River Avon empties into the English Channel by reading each "
        "page. There are several rivers with that name across Britain."
    )
    assert extract_named_candidates(prose) == []


def test_single_numbered_item_is_not_candidates():
    single = "The one option is:\n  1. River Avon, Bristol — the Bristol Avon\nNothing else."
    assert extract_named_candidates(single) == []


def test_instruction_steps_051_return_empty():
    # Imperative-verb-led numbered steps must fail open (NOT a candidate list).
    assert extract_named_candidates(INSTRUCTION_MANDATE_051) == []


def test_instruction_steps_065_return_empty():
    assert extract_named_candidates(INSTRUCTION_MANDATE_065) == []


def test_stray_prose_numbers_do_not_falsepositive():
    # Citation/percentage-style "1." occurrences mid-prose are not line-start items.
    txt = (
        "The population grew by 1.5% last year (see ref. 1. and ref. 2. below). "
        "This is a plain paragraph with no enumerated candidate list."
    )
    assert extract_named_candidates(txt) == []


def test_candidate_list_without_delimiter_uses_whole_line():
    # No em-dash/parenthetical: the whole (non-imperative) item is the name.
    mandate = (
        "Pick the correct river:\n"
        "  1. River Avon, Bristol\n"
        "  2. River Avon, Hampshire\n"
    )
    assert extract_named_candidates(mandate) == [
        "River Avon, Bristol",
        "River Avon, Hampshire",
    ]


# ---------------------------------------------------------------------------
# evaluate_candidate_coverage
# ---------------------------------------------------------------------------


def _visit_node(graph, page_title):
    """Attach a node carrying a SUCCESSFUL visit action_result for ``page_title``.

    A candidate is only credited as resolved when a real page was OPENED for it — a
    bare node title (or a search-result snippet) must NOT count, or a weak model could
    satisfy the gate without ever reading the disambiguating infobox fact.
    """
    graph.add_child(
        graph.root_id(),
        title="visit node",
        details={
            DetailKey.ACTION_RESULT.value: {
                "action": IdeaActionType.VISIT.value,
                "success": True,
                "page_title": f"{page_title} - Wikipedia",
                "url": "https://en.wikipedia.org/wiki/x",
                "content": "infobox mouth: some body of water",
            }
        },
    )


def _graph_with_visits(page_titles):
    graph = IdeaDag(root_title="root", root_details={"mandate": AVON_MANDATE})
    for t in page_titles:
        _visit_node(graph, t)
    return graph


def test_coverage_satisfied_when_all_candidates_visited():
    graph = _graph_with_visits(
        [
            "River Avon, Bristol",
            "River Avon, Warwickshire",
            "River Avon, Hampshire",
            "River Avon, Strathspey",
        ]
    )
    res = evaluate_candidate_coverage(graph, AVON_MANDATE)
    assert isinstance(res, CandidateCoverageResult)
    assert res.satisfied is True
    assert res.missing == []
    assert len(res.resolved) == 4


def test_coverage_unsatisfied_reports_missing():
    graph = _graph_with_visits(
        ["River Avon, Bristol", "River Avon, Warwickshire"]
    )
    res = evaluate_candidate_coverage(graph, AVON_MANDATE)
    assert res.satisfied is False
    assert res.missing == ["River Avon, Hampshire", "River Avon, Strathspey"]
    assert set(res.resolved) == {"River Avon, Bristol", "River Avon, Warwickshire"}


def test_coverage_matches_against_visited_page_title():
    # A candidate is resolved by a successful VISIT action_result's page_title.
    graph = _graph_with_visits(
        ["River Avon, Bristol", "River Avon, Warwickshire", "River Avon, Strathspey"]
    )
    graph.add_child(
        graph.root_id(),
        title="visit node",
        details={
            DetailKey.ACTION_RESULT.value: {
                "action": IdeaActionType.VISIT.value,
                "success": True,
                "page_title": "River Avon, Hampshire",
            }
        },
    )
    res = evaluate_candidate_coverage(graph, AVON_MANDATE)
    assert res.satisfied is True


def test_bare_node_titles_do_not_resolve_candidates():
    # The root's title IS the mandate (which enumerates every candidate) and plain
    # thought/plan node titles are not page reads: neither may satisfy the gate.
    graph = IdeaDag(root_title="root", root_details={"mandate": AVON_MANDATE})
    for name in (
        "River Avon, Bristol",
        "River Avon, Warwickshire",
        "River Avon, Hampshire",
        "River Avon, Strathspey",
    ):
        graph.add_child(graph.root_id(), title=name)
    res = evaluate_candidate_coverage(graph, AVON_MANDATE)
    assert res.satisfied is False
    assert res.resolved == []
    assert set(res.missing) == {
        "River Avon, Bristol",
        "River Avon, Warwickshire",
        "River Avon, Hampshire",
        "River Avon, Strathspey",
    }


def test_search_results_do_not_resolve_candidates():
    # Search returns only engine snippets that MENTION a candidate's name without ever
    # reading its criterion — this must NOT credit coverage (the original loophole).
    graph = IdeaDag(root_title="root", root_details={"mandate": AVON_MANDATE})
    graph.add_child(
        graph.root_id(),
        title="search node",
        details={
            DetailKey.ACTION_RESULT.value: {
                "action": IdeaActionType.SEARCH.value,
                "success": True,
                "results": [
                    {"title": "River Avon, Bristol - Wikipedia", "url": "u1"},
                    {"title": "River Avon, Warwickshire - Wikipedia", "url": "u2"},
                    {"title": "River Avon, Hampshire - Wikipedia", "url": "u3"},
                    {"title": "River Avon, Strathspey - Wikipedia", "url": "u4"},
                ],
                "content": "River Avon, Bristol River Avon, Warwickshire "
                "River Avon, Hampshire River Avon, Strathspey",
            }
        },
    )
    res = evaluate_candidate_coverage(graph, AVON_MANDATE)
    assert res.satisfied is False
    assert res.resolved == []


def test_failed_visit_does_not_resolve_candidate():
    graph = IdeaDag(root_title="root", root_details={"mandate": AVON_MANDATE})
    for name in ("River Avon, Bristol", "River Avon, Warwickshire", "River Avon, Hampshire"):
        _visit_node(graph, name)
    graph.add_child(
        graph.root_id(),
        title="failed visit",
        details={
            DetailKey.ACTION_RESULT.value: {
                "action": IdeaActionType.VISIT.value,
                "success": False,
                "page_title": "River Avon, Strathspey - Wikipedia",
            }
        },
    )
    res = evaluate_candidate_coverage(graph, AVON_MANDATE)
    assert res.satisfied is False
    assert res.missing == ["River Avon, Strathspey"]


def test_coverage_fail_open_when_no_candidates():
    graph = _graph_with_visits(["anything"])
    res = evaluate_candidate_coverage(graph, INSTRUCTION_MANDATE_051)
    assert res.satisfied is True
    assert res.named == []
    assert res.missing == []


# ---------------------------------------------------------------------------
# Scope: the gate is NOT branch-eliminate-only (pins the corrected docstring)
# ---------------------------------------------------------------------------

def test_gate_engages_for_a_breadth_fanout_roster_not_only_branch_eliminate():
    """test_052 (canonical 6-way fan-out) enumerates six NOVELS, so the gate fires.

    The docstring used to claim this gate "stays inert for chain / parallel_merge /
    plain fan-out tasks"; that was false. Pin the real behaviour: an enumerated roster
    of >= 2 names engages the gate whatever the task shape, and a run that opened a page
    for only some of them is unsatisfied.
    """
    import importlib

    mandate = importlib.import_module(
        "agent.app.idea_tests.test_052_tier5_breadth_aggregation"
    ).get_task_statement()

    assert extract_named_candidates(mandate) == [
        "Pride and Prejudice",
        "Crime and Punishment",
        "Mrs Dalloway",
        "The Great Gatsby",
        "The Old Man and the Sea",
        "Beloved",
    ]

    graph = IdeaDag(root_title="root", root_details={"mandate": mandate})
    # Two author pages opened; each mentions its own novel, none mentions the other four.
    _visit_node(graph, "Jane Austen — author of Pride and Prejudice")
    _visit_node(graph, "Fyodor Dostoevsky — author of Crime and Punishment")

    res = evaluate_candidate_coverage(graph, mandate)
    assert res.satisfied is False
    assert res.resolved == ["Pride and Prejudice", "Crime and Punishment"]
    assert res.missing == [
        "Mrs Dalloway",
        "The Great Gatsby",
        "The Old Man and the Sea",
        "Beloved",
    ]


def test_gate_still_inert_for_a_chain_mandate():
    # The half of the old docstring that WAS true: a numbered INSTRUCTION list (chain)
    # yields no roster, so the gate imposes nothing.
    graph = _graph_with_visits(["anything"])
    assert evaluate_candidate_coverage(graph, INSTRUCTION_MANDATE_065).satisfied is True


def test_strip_enumerated_items_blanks_list_lines_only():
    stripped = strip_enumerated_items(AVON_MANDATE)
    assert "River Avon, Bristol" not in stripped
    assert "eliminate to one survivor" in stripped
    assert "empties into the ENGLISH CHANNEL" in stripped
    assert strip_enumerated_items("") == ""
