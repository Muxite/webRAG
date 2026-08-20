"""The automatic pre-expansion short-circuit — a retrieved plan REPLACING an invented one.

``_handle_expansion_node`` normally asks the expansion policy (an LLM) to invent a
decomposition. With ``plan_library_enabled`` + ``plan_library_auto_enabled`` on, a confidently
matched pre-authored template's filled leaves become the expansion instead, and the LLM is
never called. This file pins the five properties that mechanism lives or dies by:

* **flag-off is inert** — no ``PlanLibrary`` is constructed, no Chroma call is attempted, and
  the expansion policy runs exactly as before (stubs that raise if touched prove the zero-I/O
  claim, matching the A1-A5 flag-off discipline);
* **a hit really short-circuits** — ``expansion.expand()`` is not called at all, the children
  come from the template, the parent carries the template's aggregation guidance, and a
  chain-shaped template's ``depends_on`` becomes real ``requires_data`` node references;
* **every failure falls through** — no match, a low-confidence match, a slot-fill failure, and
  an outright exception all land back on the organic path rather than degrading the run;
* **the two logs stay separate and fire once each** — the contract log (did the leaf deliver)
  and the retrieval log (was the match right) are independently env-gated;
* **origin attribution is real** — a library-sourced leaf's ``contract_made`` row says
  ``plan_library_auto`` and names the template, which is the join key the whole dogfooding
  design rests on.

Everything here is offline: a temp template corpus on disk, a ``ConnectorChroma``-shaped fake
(``plan_library_retrieval_test``'s pattern) and an ``AgentIO`` stub with no LLM at all — the
seed slots are deterministic, so the fill is exercised end to end without a model.
"""
from __future__ import annotations

import json
import logging

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies import BestScoreSelectionPolicy, SimpleMergePolicy
from agent.app.idea_policies import plan_library as adapter
from agent.app.idea_policies.base import (
    DetailKey,
    DecompositionPolicy,
    EvaluationPolicy,
    ExpansionPolicy,
    IdeaActionType,
    IdeaNodeStatus,
)
from agent.app.idea_policies.post_expansion_hooks import (
    GroundingEvidenceEnforcementHook,
    MandatePhraseEnforcementHook,
)
from agent.app.plan_library import retrieval as R
from agent.app.plan_library import retrieval_log
from agent.app.plan_library.schema import TemplateValidationError
from agent.app.testing import contract_log


# The enumerated candidate list is LINE-oriented: ``extract_named_candidates`` matches at line
# starts, so this mandate only fills the template's ENTITY_LIST slot while its newlines are
# intact. That is exactly why the engine must hand slot-fill the RAW mandate and not the
# whitespace-collapsed retrieval query — see the raw-mandate test at the bottom.
_MANDATE = (
    "Of the following three peaks, which has the greatest topographic prominence?\n"
    "1. Kongur Tagh (China)\n"
    "2. Noshaq (Afghanistan)\n"
    "3. Distaghil Sar (Pakistan)\n"
    "Open each peak's own page and read the value; do not guess from memory."
)
_FIELD = "topographic prominence in metres"
_PEAKS = ["Kongur Tagh", "Noshaq", "Distaghil Sar"]

_ARGMAX_TEMPLATE = {
    "template_id": "argmax_t",
    "archetype": "argmax",
    "title": "argmax over N page reads",
    "embedding_text": "which of these candidates has the greatest value for one page field",
    "provenance": {"source": "hand_authored", "based_on_tasks": ["062"]},
    "slots": [
        {
            "name": "candidates",
            "kind": "entity_list",
            "extraction": "regex_candidate_list",
            "min_arity": 2,
            "description": "the enumerated candidates the mandate lists",
        },
        {
            "name": "field",
            "kind": "field",
            "extraction": "default",
            "default": _FIELD,
            "description": "the quantity to read on each candidate's page",
        },
    ],
    "leaves": [
        {
            "id_pattern": "<<item.key>>_field",
            "for_each": "candidates",
            "instruction": (
                "Open the authoritative page for <<item.name>> and read its <<field>> "
                "directly from that page. Report ONLY that value and the source URL."
            ),
            "expect": "<<item.name>>: its <<field>> -- source URL",
        },
    ],
    "aggregation": "Write out every candidate's <<field>> in full before naming the winner.",
}

_CHAIN_TEMPLATE = {
    "template_id": "chain_t",
    "archetype": "chain",
    "title": "N-hop entity chain",
    "embedding_text": "follow a research chain where each hop is revealed by the previous page",
    "provenance": {"source": "hand_authored", "based_on_tasks": ["051"]},
    "slots": [
        {
            "name": "hops",
            "kind": "entity_list",
            "extraction": "regex_candidate_list",
            "min_arity": 2,
            "description": "the chain's hops in order",
        },
    ],
    "leaves": [
        {
            "id_pattern": "chain_start",
            "instruction": "Start of the chain: open the seed page and identify the next entity.",
            "expect": "the next entity in the chain -- source URL",
        },
        {
            "id_pattern": "<<item.key>>_hop",
            "for_each": "hops",
            "instruction": "Continue the chain to <<item.name>> using the previous step's result.",
            "expect": "<<item.name>>: the value read from its page -- source URL",
            "depends_on": ["chain_start", "<<item.key>>_hop"],
        },
    ],
    "aggregation": "Write the chain out link by link before reporting the final value.",
}


# --------------------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------------------


class _FakeCollection:
    def __init__(self, space="cosine"):
        self.configuration_json = {"hnsw": {"space": space}}
        self.metadata = {"hnsw:space": space}


class FakeChroma:
    """The two ``ConnectorChroma`` methods retrieval uses (``plan_library_retrieval_test``)."""

    def __init__(self, hits=()):
        self.hits = list(hits)  # [(template_id, distance, archetype), ...]
        self.queries = []

    async def get_or_create_collection(self, collection, metadata=None):
        return _FakeCollection()

    async def query_chroma(self, collection, query_texts, n_results=3, where=None):
        self.queries.append(list(query_texts))
        hits = self.hits[:n_results]
        return {
            "ids": [[h[0] for h in hits]],
            "distances": [[h[1] for h in hits]],
            "metadatas": [[{"template_id": h[0], "archetype": h[2]} for h in hits]],
        }


class ForbiddenWithFlagOff(BaseException):
    """What the flag-off stubs raise — a ``BaseException`` deliberately.

    The short-circuit's own fail-toward-silence net catches ``Exception``, so a plain
    ``AssertionError`` from a "must never be called" stub would be SWALLOWED and the flag-off
    test would pass for exactly the wrong reason (mechanism ran, blew up, fell back quietly).
    """


class ExplodingChroma:
    """Any Chroma call with the flag off is a bug."""

    def __getattr__(self, name):  # pragma: no cover - must never be reached
        raise ForbiddenWithFlagOff(f"plan library touched chroma with the flag off ({name})")


class DummyIO:
    """An ``AgentIO`` stand-in with a vector DB but deliberately NO LLM.

    ``slot_fill`` skips its (single) LLM pass when ``io`` has no ``query_llm_with_fallback``,
    so the seed templates here fill purely from their deterministic extractors — the whole
    automatic path runs end to end at zero model cost.
    """

    def __init__(self, connector_chroma=None):
        self.connector_chroma = connector_chroma

    def set_telemetry(self, telemetry):
        return None


class FakeExpansion(ExpansionPolicy):
    """The LLM-invented path. ``calls`` is the load-bearing assertion of this whole file."""

    def __init__(self, settings=None, candidates=None):
        super().__init__(settings=settings)
        self.calls = 0
        self._candidates = candidates if candidates is not None else _ORGANIC_CANDIDATES

    async def expand(self, graph, node_id, memories=None):
        self.calls += 1
        return [json.loads(json.dumps(c)) for c in self._candidates]


_ORGANIC_CANDIDATES = [
    {
        "title": "Open the Kongur Tagh page and read its prominence",
        "details": {
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.EXPECT.value: "Kongur Tagh: its prominence in metres -- source URL",
        },
    },
    {
        "title": "Search for the Noshaq page",
        "details": {DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
    },
]


class FakeEvaluation(EvaluationPolicy):
    async def evaluate(self, graph, node_id):
        graph.evaluate(node_id, 0.6)
        return 0.6

    async def evaluate_batch(self, graph, parent_id, candidate_ids):
        return {nid: (graph.evaluate(nid, 0.6) or 0.6) for nid in candidate_ids}


class FakeDecomposition(DecompositionPolicy):
    def should_decompose(self, graph, node_id):
        return False


class FakeActionRegistry:
    def __init__(self, settings):
        self.settings = dict(settings or {})

    def get(self, action_type):  # pragma: no cover - no action is executed here
        return None


def _make_engine(*, enabled=True, auto=True, io=None, candidates=None):
    settings = {
        "plan_library_enabled": enabled,
        "plan_library_auto_enabled": auto,
        "allow_unscored_selection": True,
        "min_score_threshold": 0.0,
        "best_first_global": False,
        "got_dedup_enabled": False,
        "got_embed_on_create": False,
        "auto_parallel_siblings": False,
        "semantic_dedup_visits_enabled": False,
    }
    expansion = FakeExpansion(settings, candidates=candidates)
    engine = IdeaDagEngine(
        io=io if io is not None else DummyIO(ExplodingChroma()),
        settings=settings,
        expansion=expansion,
        evaluation=FakeEvaluation(settings),
        selection=BestScoreSelectionPolicy(settings=settings),
        decomposition=FakeDecomposition(settings),
        merge=SimpleMergePolicy(settings=settings),
        actions=FakeActionRegistry(settings),
        post_expansion_hooks=[],
    )
    return engine, expansion


def _library(tmp_path, templates=(_ARGMAX_TEMPLATE, _CHAIN_TEMPLATE)) -> R.PlanLibrary:
    for template in templates:
        (tmp_path / f"{template['template_id']}.json").write_text(
            json.dumps(template), encoding="utf-8"
        )
    return R.PlanLibrary(templates_dir=tmp_path, warn_on_drift=False)


def _graph(mandate=_MANDATE) -> IdeaDag:
    return IdeaDag(root_title=mandate, root_details={"mandate": mandate})


def _wire(engine, tmp_path, hits, templates=(_ARGMAX_TEMPLATE, _CHAIN_TEMPLATE)):
    """Give the engine a temp corpus and a scripted index — the lazy-cache seam."""
    engine._plan_library_corpus_cache = _library(tmp_path, templates)
    chroma = FakeChroma(hits)
    engine.io = DummyIO(chroma)
    return chroma


def _children(graph, node):
    return [graph.get_node(cid) for cid in node.children]


def _leaves(graph, node):
    """The template's own leaves — i.e. the search children, without the page-visit siblings
    ``link_page_visits`` follows each of them through into."""
    return [
        c for c in _children(graph, node)
        if c.details.get(DetailKey.ACTION.value) == IdeaActionType.SEARCH.value
    ]


def _visits(graph, node):
    return [
        c for c in _children(graph, node)
        if c.details.get(DetailKey.ACTION.value) == IdeaActionType.VISIT.value
    ]


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _enable_logs(monkeypatch, tmp_path):
    contract_out = tmp_path / "contract_log.jsonl"
    retrieval_out = tmp_path / "plan_retrievals.jsonl"
    monkeypatch.setenv("IDEA_TEST_CONTRACT_LOG", "1")
    monkeypatch.setenv("IDEA_TEST_CONTRACT_LOG_PATH", str(contract_out))
    monkeypatch.setenv(retrieval_log.ENV_FLAG, "1")
    monkeypatch.setenv(retrieval_log.ENV_PATH, str(retrieval_out))
    return contract_out, retrieval_out


def _disable_logs(monkeypatch, tmp_path):
    monkeypatch.delenv("IDEA_TEST_CONTRACT_LOG", raising=False)
    monkeypatch.delenv(retrieval_log.ENV_FLAG, raising=False)
    monkeypatch.setenv("IDEA_TEST_CONTRACT_LOG_PATH", str(tmp_path / "contract_log.jsonl"))
    monkeypatch.setenv(retrieval_log.ENV_PATH, str(tmp_path / "plan_retrievals.jsonl"))


@pytest.fixture(autouse=True)
def _quiet_logs(monkeypatch, tmp_path):
    """Default every test to logs OFF, so a stray row is always the test's own doing."""
    _disable_logs(monkeypatch, tmp_path)


# --------------------------------------------------------------------------------------
# flag off — zero I/O, zero behavioural change
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("enabled,auto", [(False, False), (False, True), (True, False)])
@pytest.mark.asyncio
async def test_flag_off_never_builds_a_library_or_touches_chroma(monkeypatch, enabled, auto):
    """Both switches must be on. Anything else is byte-identical to the pre-phase engine."""

    def _boom(*args, **kwargs):  # pragma: no cover - must never be called
        raise ForbiddenWithFlagOff("a PlanLibrary was constructed with the flag off")

    monkeypatch.setattr(R, "PlanLibrary", _boom)
    engine, expansion = _make_engine(enabled=enabled, auto=auto)
    graph = _graph()
    parent = graph.add_child(graph.root_id(), "parent goal", details={})

    result = await engine._handle_expansion_node(graph, parent.node_id, 0, None)

    assert result == parent.node_id
    assert expansion.calls == 1, "the LLM path still runs"
    assert len(parent.children) == len(_ORGANIC_CANDIDATES)
    assert engine._plan_library_corpus_cache is None
    for child in _children(graph, parent):
        assert adapter.PLAN_LIBRARY_ORIGIN not in child.details
        assert DetailKey.REQUIRES_DATA.value not in child.details
    assert adapter.PLAN_LIBRARY_TEMPLATE_ID not in parent.details


@pytest.mark.asyncio
async def test_flag_off_writes_no_retrieval_rows(monkeypatch, tmp_path):
    contract_out, retrieval_out = _enable_logs(monkeypatch, tmp_path)

    def _boom(*args, **kwargs):  # pragma: no cover - must never be called
        raise ForbiddenWithFlagOff("a PlanLibrary was constructed with the flag off")

    monkeypatch.setattr(R, "PlanLibrary", _boom)
    engine, expansion = _make_engine(enabled=False, auto=False)
    graph = _graph()
    parent = graph.add_child(graph.root_id(), "parent goal", details={})

    await engine._handle_expansion_node(graph, parent.node_id, 0, None)

    assert not retrieval_out.exists(), "no retrieval attempt happened, so no row exists"
    # The organic leaf's contract IS still logged — and still attributed to nobody's template.
    made = [r for r in _rows(contract_out) if r["event"] == "contract_made"]
    assert [(r["origin"], r["template_id"]) for r in made] == [("organic", None)]
    assert not [r for r in _rows(contract_out) if r["event"] == "plan_library_search"]
    assert expansion.calls == 1


def test_the_corpus_is_built_once_per_engine(monkeypatch):
    """``PlanLibrary.__init__`` parses the whole corpus off disk — once per engine, not per
    expansion step, and never at all until the first retrieval."""
    built = []

    class _Counting:
        def __init__(self, *args, **kwargs):
            built.append(self)

    monkeypatch.setattr(R, "PlanLibrary", _Counting)
    engine, _ = _make_engine()

    assert built == []  # construction is lazy
    first = engine._plan_library_corpus()
    second = engine._plan_library_corpus()
    assert first is second and len(built) == 1


# --------------------------------------------------------------------------------------
# the hit — the LLM path is genuinely skipped
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_apply_builds_the_children_from_the_template(tmp_path):
    engine, expansion = _make_engine()
    _wire(engine, tmp_path, [("argmax_t", 0.30, "argmax"), ("chain_t", 0.70, "chain")])
    graph = _graph()
    root = graph.get_node(graph.root_id())

    result = await engine._handle_expansion_node(graph, graph.root_id(), 0, None)

    assert result == graph.root_id()
    assert expansion.calls == 0, "the LLM invention path must not run at all"

    children = _leaves(graph, root)
    assert len(children) == len(_PEAKS)
    for child, peak in zip(children, _PEAKS):
        details = child.details
        assert details[adapter.PLAN_LIBRARY_ORIGIN] == adapter.ORIGIN_AUTO
        assert details[adapter.PLAN_LIBRARY_TEMPLATE_ID] == "argmax_t"
        assert details[adapter.PLAN_LIBRARY_LEAF_ID].endswith("_field")
        assert details[DetailKey.ACTION.value] == IdeaActionType.SEARCH.value
        assert details[DetailKey.QUERY.value] == f"{peak} {_FIELD}"
        assert peak in details[DetailKey.EXPECT.value] and _FIELD in details[DetailKey.EXPECT.value]
        assert details[DetailKey.IS_LEAF.value] is True
        # independent per-candidate reads: nothing to wait for
        assert DetailKey.REQUIRES_DATA.value not in details

    # the template's aggregation discipline reaches the merge through the PARENT's intent
    assert root.details[DetailKey.INTENT.value] == (
        f"Write out every candidate's {_FIELD} in full before naming the winner."
    )
    assert root.details[adapter.PLAN_LIBRARY_TEMPLATE_ID] == "argmax_t"
    assert root.details[DetailKey.EXPANSION_META.value] == {
        DetailKey.EXECUTE_ALL_CHILDREN.value: True
    }


@pytest.mark.asyncio
async def test_every_library_leaf_is_followed_through_into_its_own_page_visit(tmp_path):
    """The other half of the post-expand wiring. A template leaf is a PAGE READ, but it can
    only be emitted as a search — so without this the whole subtree's grounding is whatever
    the generic hook manages (one visit, first search, generic link_idea) and an N-way fan-out
    loses N-1 entities."""
    engine, expansion = _make_engine()
    _wire(engine, tmp_path, [("argmax_t", 0.30, "argmax")])
    graph = _graph()
    root = graph.get_node(graph.root_id())

    await engine._handle_expansion_node(graph, graph.root_id(), 0, None)

    searches, visits = _leaves(graph, root), _visits(graph, root)
    assert len(visits) == len(searches) == len(_PEAKS), "one page read per leaf, not one shared"
    for search, visit, peak in zip(searches, visits, _PEAKS):
        # each visit is fed by ITS OWN leaf's search...
        # ...and declares the `url` slot the resolved-value channel fills at dispatch: this
        # visit's page has no URL until that search runs, which is the whole point of a slot.
        assert visit.details[DetailKey.REQUIRES_DATA.value] == {
            "type": "urls_from_search",
            "source_node_id": search.node_id,
            "slot": "url",
        }
        # ...and knows which page it is looking for among that search's results
        assert peak in visit.details["link_idea"]
        assert peak in visit.details[DetailKey.EXPECT.value]
        assert visit.details[adapter.PLAN_LIBRARY_VISIT_FOR] == search.details[
            adapter.PLAN_LIBRARY_LEAF_ID
        ]
        assert visit.details[DetailKey.PARENT_GOAL.value] == search.details[
            DetailKey.PARENT_GOAL.value
        ], "the visit inherits the parent metadata the expansion threaded onto its search"


@pytest.mark.asyncio
async def test_the_generic_grounding_hooks_have_nothing_left_to_inject(tmp_path):
    """The fix is meant to make the task-blind safety net REDUNDANT here, not to fight it:
    both mandate-driven hooks are idempotent on an existing visit child, so they add nothing
    to a library-sourced subtree — while staying exactly as they are for organic plans."""
    engine, _ = _make_engine()
    _wire(engine, tmp_path, [("argmax_t", 0.30, "argmax")])
    graph = _graph()

    await engine._handle_expansion_node(graph, graph.root_id(), 0, None)

    # ...and make the first search look finished, which is the state the grounding hook waits
    # for (before this fix, that is exactly when it injected its single blind visit).
    first_search = _leaves(graph, graph.get_node(graph.root_id()))[0]
    first_search.status = IdeaNodeStatus.DONE
    first_search.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.SEARCH.value,
        "success": True,
        "results": [{"title": _PEAKS[0], "url": "https://en.wikipedia.org/wiki/Kongur_Tagh"}],
    }

    before = graph.node_count()
    for hook in (MandatePhraseEnforcementHook(), GroundingEvidenceEnforcementHook()):
        hook.apply(graph, graph.root_id(), 1, _MANDATE, logging.getLogger("test"))

    assert graph.node_count() == before
    assert not [
        n for n in _visits(graph, graph.get_node(graph.root_id()))
        if adapter.PLAN_LIBRARY_VISIT_FOR not in n.details
    ]


@pytest.mark.asyncio
async def test_a_dead_search_retires_its_visit_instead_of_spinning_the_budget(tmp_path):
    """Per-leaf visits make ``requires_data`` routine rather than rare, so the routing behind
    it has to survive a source that never delivers: ``step()`` handed control back to the
    already-FAILED source on every step, which burned the whole step budget on one dead leaf.
    Now the dependent is retired and the parent gets on with its healthy leaves."""
    engine, _ = _make_engine()
    _wire(engine, tmp_path, [("argmax_t", 0.30, "argmax")])
    graph = _graph()

    await engine._handle_expansion_node(graph, graph.root_id(), 0, None)
    root = graph.get_node(graph.root_id())
    searches, visits = _leaves(graph, root), _visits(graph, root)
    searches[0].status = IdeaNodeStatus.FAILED
    searches[0].details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.SEARCH.value, "success": False, "error": "search API down",
    }

    next_id = await engine.step(graph, graph.root_id(), 1)

    assert visits[0].status is IdeaNodeStatus.SKIPPED
    assert "never arrived" in visits[0].details[DetailKey.ACTION_ERROR.value]
    # the next leaf still gets scheduled — one dead entity does not stall the fan-out
    assert next_id == searches[1].node_id
    assert visits[1].status is IdeaNodeStatus.PENDING


@pytest.mark.asyncio
async def test_a_chain_templates_dependencies_become_real_node_references(tmp_path):
    """``depends_on`` is blueprint-level until ``graph.expand()`` mints ids — this is the
    post-expand pass that turns it into the engine's own sequential-execution gate."""
    engine, expansion = _make_engine()
    _wire(engine, tmp_path, [("chain_t", 0.30, "chain"), ("argmax_t", 0.70, "argmax")])
    graph = _graph()
    root = graph.get_node(graph.root_id())

    await engine._handle_expansion_node(graph, graph.root_id(), 0, None)

    assert expansion.calls == 0
    # the chain's own leaves; each also gets a page-visit sibling, which never joins the chain
    # (a hop waits on the SEARCH that produces its URLs, exactly as `link_dependencies` wrote it)
    start, *hops = _leaves(graph, root)
    assert start.details[adapter.PLAN_LIBRARY_LEAF_ID] == "chain_start"
    assert DetailKey.REQUIRES_DATA.value not in start.details, "the first hop waits on nobody"

    # each hop waits on the one before it — the engine's own "DEPENDENCY DETECTED" gate
    assert len(hops) == len(_PEAKS)
    sources = [start] + hops[:-1]
    for hop, source in zip(hops, sources):
        requires = hop.details[DetailKey.REQUIRES_DATA.value]
        assert requires["source_node_id"] == source.node_id
        assert requires["type"]
    # a fan-in leaf points at its DEEPEST dependency; every resolved sibling is kept for the log
    assert hops[1].details[adapter.PLAN_LIBRARY_DEPENDS_ON_NODES] == [
        start.node_id, hops[0].node_id
    ]
    # ...and a dependent batch may not be fired off in parallel
    assert root.details[DetailKey.EXPANSION_META.value] == {
        DetailKey.EXECUTE_ALL_CHILDREN.value: False
    }


def test_link_dependencies_is_a_no_op_for_organic_children():
    """Why gating the post-expand pass is safe either way: LLM-invented children carry no
    plan-library markers, so the pass has nothing to resolve and writes nothing."""
    graph = _graph()
    parent = graph.add_child(graph.root_id(), "parent goal", details={})
    children = graph.expand(parent.node_id, _ORGANIC_CANDIDATES)

    assert adapter.link_dependencies(children) == 0
    assert all(DetailKey.REQUIRES_DATA.value not in c.details for c in children)


# --------------------------------------------------------------------------------------
# every failure falls through to the LLM
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_weak_match_falls_through_to_the_expansion_policy(monkeypatch, tmp_path):
    contract_out, retrieval_out = _enable_logs(monkeypatch, tmp_path)
    engine, expansion = _make_engine()
    _wire(engine, tmp_path, [("argmax_t", 0.99, "argmax")])  # similarity 0.01
    graph = _graph()
    parent = graph.add_child(graph.root_id(), "parent goal", details={})

    await engine._handle_expansion_node(graph, parent.node_id, 0, None)

    assert expansion.calls == 1
    assert len(parent.children) == len(_ORGANIC_CANDIDATES)
    assert all(
        adapter.PLAN_LIBRARY_ORIGIN not in c.details for c in _children(graph, parent)
    )

    search = [r for r in _rows(contract_out) if r["event"] == "plan_library_search"]
    assert len(search) == 1
    assert search[0]["decision"] == R.DECISION_NO_MATCH
    assert search[0]["adopted"] is False and search[0]["adopted_template_id"] is None

    row = _rows(retrieval_out)[0]
    assert row["decision"] == R.DECISION_NO_MATCH
    assert row["slot_fill"] == "not_attempted", "a weak match never spends a slot-fill call"


@pytest.mark.asyncio
async def test_a_slot_fill_failure_downgrades_to_the_organic_path(monkeypatch, tmp_path):
    """The template matches, but this mandate has no enumerated candidate list for its
    required ENTITY_LIST slot — fail toward silence, never a half-filled subtree."""
    _, retrieval_out = _enable_logs(monkeypatch, tmp_path)
    engine, expansion = _make_engine()
    _wire(engine, tmp_path, [("argmax_t", 0.30, "argmax")])
    graph = _graph("Which is the more prominent peak overall? Read the pages, do not guess.")
    parent = graph.add_child(graph.root_id(), "parent goal", details={})

    await engine._handle_expansion_node(graph, parent.node_id, 0, None)

    assert expansion.calls == 1
    assert len(parent.children) == len(_ORGANIC_CANDIDATES)
    row = _rows(retrieval_out)[0]
    assert row["decision"] == R.DECISION_NO_MATCH  # downgraded post-hoc
    assert row["slot_fill"] == "failed" and row["applied_template_id"] is None
    assert "candidates" in row["error"]


@pytest.mark.asyncio
async def test_a_raising_library_never_breaks_the_expansion(tmp_path):
    """Even an outright exception (a broken template, a bad connector) is silence, not a crash."""

    class _Raising:
        async def retrieve(self, *args, **kwargs):
            raise TemplateValidationError("boom")

    engine, expansion = _make_engine()
    engine._plan_library_corpus_cache = _Raising()
    graph = _graph()
    parent = graph.add_child(graph.root_id(), "parent goal", details={})

    result = await engine._handle_expansion_node(graph, parent.node_id, 0, None)

    assert (result, expansion.calls) == (parent.node_id, 1)
    assert len(parent.children) == len(_ORGANIC_CANDIDATES)


@pytest.mark.asyncio
async def test_a_raising_fill_also_falls_through(tmp_path):
    """``fill_from_query`` swallows its own failures; this pins the engine's outer net too."""

    class _RaisingFill:
        def __init__(self, inner):
            self._inner = inner

        async def retrieve(self, *args, **kwargs):
            return await self._inner.retrieve(*args, **kwargs)

        async def fill_from_query(self, *args, **kwargs):
            raise TemplateValidationError("template exploded while filling")

    engine, expansion = _make_engine()
    _wire(engine, tmp_path, [("argmax_t", 0.30, "argmax")])
    engine._plan_library_corpus_cache = _RaisingFill(engine._plan_library_corpus_cache)
    graph = _graph()

    await engine._handle_expansion_node(graph, graph.root_id(), 0, None)

    assert expansion.calls == 1
    root = graph.get_node(graph.root_id())
    assert len(root.children) == len(_ORGANIC_CANDIDATES)
    assert adapter.PLAN_LIBRARY_TEMPLATE_ID not in root.details


# --------------------------------------------------------------------------------------
# instrumentation — two independent streams, one row each
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_both_logs_fire_exactly_once_on_a_hit(monkeypatch, tmp_path):
    contract_out, retrieval_out = _enable_logs(monkeypatch, tmp_path)
    engine, _ = _make_engine()
    _wire(engine, tmp_path, [("argmax_t", 0.30, "argmax"), ("chain_t", 0.70, "chain")])
    graph = _graph()

    await engine._handle_expansion_node(graph, graph.root_id(), 0, None)

    search = [r for r in _rows(contract_out) if r["event"] == "plan_library_search"]
    assert len(search) == 1
    assert search[0]["mode"] == R.CALL_SITE_AUTO
    assert search[0]["decision"] == R.DECISION_AUTO_APPLY
    assert search[0]["adopted"] is True
    assert search[0]["adopted_template_id"] == "argmax_t"
    assert search[0]["node_id"] == graph.root_id()
    assert [m["template_id"] for m in search[0]["matches"]] == ["argmax_t", "chain_t"]

    rows = _rows(retrieval_out)
    assert len(rows) == 1
    assert rows[0]["call_site"] == R.CALL_SITE_AUTO
    assert rows[0]["applied_template_id"] == "argmax_t" and rows[0]["slot_fill"] == "filled"
    assert rows[0]["node_id"] == graph.root_id()
    assert len(rows[0]["retrieval_id"]) == 32, "the uuid4 join key back to the contract log"
    # the second stream's extra payload: WHAT the template was instantiated with, for review
    assert [c["name"] for c in rows[0]["slot_values"]["candidates"]] == _PEAKS
    assert rows[0]["slot_values"]["field"] == _FIELD


@pytest.mark.asyncio
async def test_each_log_is_gated_independently(monkeypatch, tmp_path):
    """One flag on, one off: neither stream depends on the other being enabled."""
    contract_out = tmp_path / "contract_log.jsonl"
    retrieval_out = tmp_path / "plan_retrievals.jsonl"
    monkeypatch.setenv("IDEA_TEST_CONTRACT_LOG_PATH", str(contract_out))
    monkeypatch.setenv(retrieval_log.ENV_PATH, str(retrieval_out))
    monkeypatch.setenv(retrieval_log.ENV_FLAG, "1")
    monkeypatch.delenv("IDEA_TEST_CONTRACT_LOG", raising=False)
    engine, _ = _make_engine()
    _wire(engine, tmp_path, [("argmax_t", 0.30, "argmax")])
    graph = _graph()

    await engine._handle_expansion_node(graph, graph.root_id(), 0, None)

    assert not contract_out.exists()
    assert len(_rows(retrieval_out)) == 1


@pytest.mark.asyncio
async def test_contract_made_is_attributed_to_the_template_that_authored_the_leaf(
    monkeypatch, tmp_path,
):
    """The dogfooding join: a library-sourced leaf's contract outcome must name its template,
    not the ``organic`` placeholder every LLM-invented leaf gets."""
    contract_out, _ = _enable_logs(monkeypatch, tmp_path)
    engine, _ = _make_engine()
    _wire(engine, tmp_path, [("argmax_t", 0.30, "argmax")])
    graph = _graph()

    await engine._handle_expansion_node(graph, graph.root_id(), 0, None)

    made = [r for r in _rows(contract_out) if r["event"] == "contract_made"]
    assert len(made) == len(_PEAKS)
    for row, peak in zip(made, _PEAKS):
        assert row["origin"] == adapter.ORIGIN_AUTO == "plan_library_auto"
        assert row["template_id"] == "argmax_t"
        assert peak in row["expect"]
        assert row["ancestor_ids"] == [graph.root_id()]


# --------------------------------------------------------------------------------------
# the raw mandate — the argument that silently breaks every automatic fill if wrong
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slot_fill_receives_the_raw_mandate_not_the_collapsed_query(tmp_path):
    """``retrieve()`` whitespace-collapses its query text, and the deterministic
    candidate-list extractor matches at LINE starts — so handing slot-fill the query text
    instead of the raw mandate would quietly reduce every enumerated list to nothing."""
    seen = {}

    class _Spy:
        def __init__(self, inner):
            self._inner = inner

        async def retrieve(self, *args, **kwargs):
            result = await self._inner.retrieve(*args, **kwargs)
            seen["query_text"] = result.query_text
            return result

        async def fill_from_query(self, result, **kwargs):
            seen["mandate"] = kwargs.get("mandate")
            return await self._inner.fill_from_query(result, **kwargs)

    engine, expansion = _make_engine()
    _wire(engine, tmp_path, [("argmax_t", 0.30, "argmax")])
    engine._plan_library_corpus_cache = _Spy(engine._plan_library_corpus_cache)
    graph = _graph()

    await engine._handle_expansion_node(graph, graph.root_id(), 0, None)

    assert seen["mandate"] == _MANDATE
    assert "\n" in seen["mandate"], "the enumeration only survives with line structure intact"
    assert "\n" not in seen["query_text"], "...which the collapsed query text no longer has"
    # and the proof it mattered: the fill succeeded, off the deterministic extractor alone
    assert expansion.calls == 0
    assert len(_leaves(graph, graph.get_node(graph.root_id()))) == len(_PEAKS)
