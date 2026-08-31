"""
Test 167: N-SWEEP member N=16 -- roster discovery -> 16-way independent fan-out -> argmin
Level: graph   Weight: long   Difficulty: 8/10   Category: N-sweep Breadth Aggregation

One of four tasks (165 N=4, 166 N=8, 167 N=16, 168 N=32) built from ONE template in
``nsweep_toho_runtime_family``, where the roster size is the ONLY variable: the four rosters
are nested prefixes of the same ordered list, so item i is literally the same page read in
every member of the family and per-item difficulty cannot drift with N.

  ROSTER (discovered, one page)  'List of Godzilla films' numbers the Japanese (Toho) films in
    release order. The mandate names only that page and the prefix length 16; no film title
    appears in it, so the roster must be fetched before any fan-out can begin.
  FAN-OUT (16 independent one-page reads)  each film's own article -> infobox running time.
  MERGE  argmin over the 16 runtimes, plus the combined total (wrong if one item is missing).

Ground truth (en.wikipedia infobox ``runtime``, verified live 2026-08-30 -- full 32-row table,
provenance, margins and the Wikidata cross-check that it FAILED are in the family module):

  shortest of the first 16: All Monsters Attack -- 70 minutes   <-- KEYSTONE
  runner-up: 81 minutes  ->  margin 11 minutes
  combined total running time of the 16 films: 1405 minutes

Decision-critical item: #10 (the argmin). Dropping it flips the answer; every other item is
argmin-irrelevant but still moves the total. Sources: https://en.wikipedia.org/wiki/List_of_Godzilla_films plus the
16 linked film articles.
"""

from typing import Any, Callable, Dict, List

from agent.app.idea_tests import nsweep_toho_runtime_family as family


N = 16

ROSTER = family.roster(N)
KEYSTONE = family.keystone(N)
TOTAL_RUNTIME = family.total_runtime(N)


def get_test_metadata() -> Dict[str, Any]:
    """Return the suite-standard metadata block for this task.

    :returns: dict with ``test_id``, ``test_name``, ``difficulty_level``, ``category``,
        ``level`` (``"graph"``) and ``weight`` (``"long"``, identical at every N so the
        budget overlay cannot become a second variable).
    """
    return {
        "test_id": "167",
        "test_name": "N-sweep N=16: roster discovery + 16-way fan-out -> shortest running time",
        "difficulty_level": "8/10",
        "category": "N-sweep Breadth Aggregation",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    """Return the mandate (family wording, N=16).

    :returns: the task statement string handed to the agent.
    """
    return family.get_task_statement(N)


def get_required_deliverables() -> List[str]:
    """Return the deliverables a complete answer must contain.

    :returns: list of deliverable descriptions.
    """
    return family.get_required_deliverables(N)


def get_success_criteria() -> List[str]:
    """Return the human-readable success criteria for this task.

    :returns: list of criteria strings.
    """
    return family.get_success_criteria(N)


def get_validation_functions() -> List[Callable]:
    """Return the validators: visit effort, the 0/1 keystone, the un-gated per-item and
    positional recall diagnostics, then the keystone-gated total and citations.

    :returns: list of validator callables.
    """
    return family.get_validation_functions(N)


def get_llm_validation_function() -> Callable:
    """No LLM judge -- every check here is deterministic.

    :returns: ``None``.
    """
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Return the offline-authored DAG (schema v2): one roster leaf + 16 positional film
    leaves templated off ``{roster}``. Leaks no title, runtime, total or answer.

    :returns: plan dict with ``leaves`` and ``aggregation``.
    """
    return family.get_compiled_plan(N)
