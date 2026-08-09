"""The on-demand plan-library search — the model ASKING for a strategy, and getting one.

Phase 4 gave the engine an automatic pre-expansion short-circuit. This is the other call
site: a ``plan_library_search`` leaf action the model may choose itself, which ranks the
template corpus, binds the winner to this mandate's entities, and REPORTS what it found —
read-only, like every other ``LeafAction``. Turning an adopted template into real graph
children is the engine's job (``_maybe_plan_library_reexpand``), because that is where every
re-expansion's termination bookkeeping already lives.

The properties pinned here:

* **the enum grew by exactly one, and the registry resolves it** — the closed-enum surface is
  the one risky part of this phase;
* **flag-off is inert** — ``plan_library_search`` is not in ``allowed_actions``, and a node
  that somehow carries the action still falls back to THINK through the PRE-EXISTING
  unknown-action gate, with no library, no Chroma and no children;
* **flag-on exposes the action to BOTH readers** of ``allowed_actions`` — the engine's
  dispatch gate and ``LlmExpansionPolicy``'s prompt menu (which snapshots settings at
  construction time, so the single patch has to land before it is built);
* **the result dict is the rebuild input** — it carries matches/adopted/adopted_template_id
  plus the bound slot values, and survives the sanitization every action result goes through;
* **adopted=True really grows the plan** — under the real engine, with the real step() routing
  descending into the new children on the next call, and WITHOUT a second Chroma query;
* **adopted=False is just a completed leaf** — no children, no marker, no re-expansion.

Everything is offline: a temp template corpus on disk, a ``ConnectorChroma``-shaped fake, and
an ``AgentIO`` stub with no LLM at all (the seed slots fill deterministically).
"""
from __future__ import annotations

import json

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_node_state import sanitize_action_result
from agent.app.idea_policies import BestScoreSelectionPolicy, SimpleMergePolicy
from agent.app.idea_policies import plan_library as adapter
from agent.app.idea_policies.actions import (
    LeafAction,
    LeafActionRegistry,
    PlanLibrarySearchLeafAction,
)
from agent.app.idea_policies.action_constants import ActionResultBuilder
from agent.app.idea_policies.base import (
    DecompositionPolicy,
    DetailKey,
    EvaluationPolicy,
    ExpansionPolicy,
    IdeaActionType,
    IdeaNodeStatus,
)
from agent.app.plan_library import retrieval as R
from agent.app.plan_library import retrieval_log
from agent.app.testing import contract_log

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
    """A ``BaseException`` on purpose: the action's own fail-toward-silence net catches
    ``Exception``, so a plain ``AssertionError`` would be swallowed and a flag-off test would
    pass for exactly the wrong reason."""


class ExplodingChroma:
    """Any Chroma call with the flag off is a bug."""

    def __getattr__(self, name):  # pragma: no cover - must never be reached
        raise ForbiddenWithFlagOff(f"plan library touched chroma with the flag off ({name})")


class DummyIO:
    """An ``AgentIO`` stand-in with a vector DB but deliberately NO LLM, so slot-fill runs on
    its deterministic extractors alone (``slot_fill`` skips its single LLM pass when ``io``
    has no ``query_llm_with_fallback``)."""

    def __init__(self, connector_chroma=None):
        self.connector_chroma = connector_chroma

    def set_telemetry(self, telemetry):
        return None


class FakeExpansion(ExpansionPolicy):
    async def expand(self, graph, node_id, memories=None):  # pragma: no cover - never invoked
        raise ForbiddenWithFlagOff("the LLM expansion path must not run in this file")


class FakeEvaluation(EvaluationPolicy):
    async def evaluate(self, graph, node_id):
        graph.evaluate(node_id, 0.6)
        return 0.6

    async def evaluate_batch(self, graph, parent_id, candidate_ids):
        return {nid: (graph.evaluate(nid, 0.6) or 0.6) for nid in candidate_ids}


class FakeDecomposition(DecompositionPolicy):
    def should_decompose(self, graph, node_id):
        return False


class RecordingSearchAction(LeafAction):
    """Stands in for the real ``search`` leaf so the children a template authored can actually
    run — ``executed`` is how this file proves the ENGINE descended into them."""

    name = "search"
    executed = []

    async def execute(self, graph, node_id, io):
        RecordingSearchAction.executed.append(node_id)
        node = graph.get_node(node_id)
        return ActionResultBuilder.success(
            action=IdeaActionType.SEARCH.value,
            node_id=node_id,
            results=[{"url": "https://example.org/peak", "title": node.title}],
        )


def _settings(*, enabled=True, action=True):
    return {
        "plan_library_enabled": enabled,
        "plan_library_action_enabled": action,
        "plan_library_auto_enabled": False,
        "allow_unscored_selection": True,
        "min_score_threshold": 0.0,
        "best_first_global": False,
        "got_dedup_enabled": False,
        "got_embed_on_create": False,
        "auto_parallel_siblings": False,
        "semantic_dedup_visits_enabled": False,
    }


def _make_engine(*, enabled=True, action=True, io=None):
    settings = _settings(enabled=enabled, action=action)
    return IdeaDagEngine(
        io=io if io is not None else DummyIO(ExplodingChroma()),
        settings=settings,
        expansion=FakeExpansion(settings),
        evaluation=FakeEvaluation(settings),
        selection=BestScoreSelectionPolicy(settings=settings),
        decomposition=FakeDecomposition(settings),
        merge=SimpleMergePolicy(settings=settings),
        post_expansion_hooks=[],
    )


def _library(tmp_path, templates=(_ARGMAX_TEMPLATE,)) -> R.PlanLibrary:
    for template in templates:
        (tmp_path / f"{template['template_id']}.json").write_text(
            json.dumps(template), encoding="utf-8"
        )
    return R.PlanLibrary(templates_dir=tmp_path, warn_on_drift=False)


def _graph(mandate=_MANDATE) -> IdeaDag:
    return IdeaDag(root_title=mandate, root_details={"mandate": mandate})


def _search_node(graph):
    return graph.add_child(
        graph.root_id(),
        "Find a proven strategy for comparing these peaks",
        details={
            DetailKey.ACTION.value: IdeaActionType.PLAN_LIBRARY_SEARCH.value,
            DetailKey.IS_LEAF.value: True,
        },
    )


def _wire(engine, monkeypatch, tmp_path, hits):
    """A temp corpus + a scripted index, for BOTH readers: the action builds its own library
    (monkeypatched constructor) and the engine's rebuild uses its cached one."""
    library = _library(tmp_path)
    monkeypatch.setattr(R, "PlanLibrary", lambda *args, **kwargs: library)
    engine._plan_library_corpus_cache = library
    chroma = FakeChroma(hits)
    engine.io = DummyIO(chroma)
    return chroma


def _children(graph, node):
    return [graph.get_node(cid) for cid in node.children]


def _searches(graph, node):
    """The template's own leaves, without the page-visit siblings each is followed through
    into (``plan_library.link_page_visits``)."""
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


@pytest.fixture(autouse=True)
def _quiet_logs(monkeypatch, tmp_path):
    """Default every test to logs OFF, so a stray row is always the test's own doing."""
    monkeypatch.delenv("IDEA_TEST_CONTRACT_LOG", raising=False)
    monkeypatch.delenv(retrieval_log.ENV_FLAG, raising=False)
    monkeypatch.setenv("IDEA_TEST_CONTRACT_LOG_PATH", str(tmp_path / "contract_log.jsonl"))
    monkeypatch.setenv(retrieval_log.ENV_PATH, str(tmp_path / "plan_retrievals.jsonl"))
    RecordingSearchAction.executed = []


# --------------------------------------------------------------------------------------
# the enum + the registry
# --------------------------------------------------------------------------------------


def test_the_action_type_enum_grew_by_exactly_one():
    """The closed-enum surface: a 7th member, and nothing else moved. Anything constructing
    settings WITHOUT `allowed_actions` falls back to this list, so its shape is load-bearing
    well beyond the plan library."""
    assert [a.value for a in IdeaActionType] == [
        "think", "search", "visit", "save", "merge", "verify", "plan_library_search",
    ]
    assert IdeaActionType("plan_library_search") is IdeaActionType.PLAN_LIBRARY_SEARCH


def test_the_registry_resolves_the_action_by_enum_and_by_name():
    registry = LeafActionRegistry(settings={})
    for key in (IdeaActionType.PLAN_LIBRARY_SEARCH, "plan_library_search"):
        assert isinstance(registry.get(key), PlanLibrarySearchLeafAction)
    assert registry.has(IdeaActionType.PLAN_LIBRARY_SEARCH)
    assert PlanLibrarySearchLeafAction.name in registry.names()


# --------------------------------------------------------------------------------------
# flag off — invisible to the model, inert in the engine
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("enabled,action", [(False, False), (False, True), (True, False)])
def test_flag_off_leaves_the_action_menu_untouched(enabled, action):
    """Both switches must be on. The menu the model sees is otherwise byte-identical."""
    engine = _make_engine(enabled=enabled, action=action)
    assert IdeaActionType.PLAN_LIBRARY_SEARCH.value not in engine.settings["allowed_actions"]
    assert engine.settings["allowed_actions"] == [
        "search", "visit", "save", "think", "merge", "verify",
    ]


@pytest.mark.asyncio
async def test_flag_off_falls_back_to_think_via_the_pre_existing_gate(monkeypatch):
    """A node hand-carrying the action with the flag off must behave exactly as it did before
    this phase: `_execute_action`'s unknown-action fallback (unchanged) routes it to THINK, and
    nothing in the library is constructed or queried."""

    def _boom(*args, **kwargs):  # pragma: no cover - must never be called
        raise ForbiddenWithFlagOff("a PlanLibrary was constructed with the flag off")

    monkeypatch.setattr(R, "PlanLibrary", _boom)
    engine = _make_engine(enabled=False, action=False)
    graph = _graph()
    node = _search_node(graph)

    result = await engine._execute_action(graph, graph.root_id(), node.node_id)

    assert result["action"] == IdeaActionType.THINK.value
    assert result["success"] is True
    assert engine._plan_library_corpus_cache is None
    assert not node.children


@pytest.mark.asyncio
async def test_flag_off_never_reexpands_even_on_an_adopted_result(monkeypatch):
    """Belt and braces: a result claiming ``adopted`` cannot grow children while unarmed."""

    def _boom(*args, **kwargs):  # pragma: no cover - must never be called
        raise ForbiddenWithFlagOff("the corpus was read with the flag off")

    monkeypatch.setattr(R, "PlanLibrary", _boom)
    engine = _make_engine(enabled=True, action=False)
    graph = _graph()
    node = _search_node(graph)
    node.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.PLAN_LIBRARY_SEARCH.value,
        "success": True,
        "adopted": True,
        "adopted_template_id": "argmax_t",
        "slot_values": {"candidates": [{"name": p, "key": p.lower()} for p in _PEAKS]},
    }

    assert await engine._maybe_plan_library_reexpand(graph, node.node_id, 0) is False
    assert not node.children
    assert "_got_reexpanded" not in node.details


# --------------------------------------------------------------------------------------
# flag on — one patch, both readers
# --------------------------------------------------------------------------------------


def test_the_single_allowed_actions_patch_reaches_both_readers():
    """`_execute_action`'s dispatch gate and `LlmExpansionPolicy`'s prompt menu read different
    dicts: the policy MERGES its settings into a new one at construction time rather than
    holding a live reference, so the patch is only covered because it lands before the policy
    is built. Pin both, since a reordering in `__init__` would silently hide the action from
    the model while leaving it dispatchable."""
    engine = IdeaDagEngine(io=DummyIO(ExplodingChroma()), settings=_settings())

    name = IdeaActionType.PLAN_LIBRARY_SEARCH.value
    assert engine.settings["allowed_actions"][-1] == name          # dispatch gate
    assert name in engine.expansion.settings["allowed_actions"]    # prompt menu

    # ...and the reason the ORDER inside `__init__` matters, demonstrated rather than asserted:
    # the policy holds a snapshot dict, so a patch applied after it was built is invisible to it.
    engine.settings["allowed_actions"] = ["think"]
    assert engine.expansion.settings is not engine.settings
    assert name in engine.expansion.settings["allowed_actions"]


def test_the_patch_never_mutates_the_callers_own_list():
    caller_list = ["search", "think"]
    settings = dict(_settings(), allowed_actions=caller_list)
    engine = IdeaDagEngine(io=DummyIO(ExplodingChroma()), settings=settings)

    assert caller_list == ["search", "think"], "the caller's list is not appended to in place"
    assert engine.settings["allowed_actions"] == [
        "search", "think", IdeaActionType.PLAN_LIBRARY_SEARCH.value,
    ]


# --------------------------------------------------------------------------------------
# execute() — a read-only report, shaped for the rebuild
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_reports_the_match_and_survives_sanitization(monkeypatch, tmp_path):
    engine = _make_engine()
    chroma = _wire(engine, monkeypatch, tmp_path, [("argmax_t", 0.30, "argmax")])
    graph = _graph()
    node = _search_node(graph)

    action = LeafActionRegistry(settings=engine.settings).get(IdeaActionType.PLAN_LIBRARY_SEARCH)
    result = await action.execute(graph, node.node_id, engine.io)

    assert result["action"] == "plan_library_search" and result["success"] is True
    assert result["decision"] == R.DECISION_AUTO_APPLY
    assert result["adopted"] is True and result["adopted_template_id"] == "argmax_t"
    assert [m["template_id"] for m in result["matches"]] == ["argmax_t"]
    assert result["matches"][0]["similarity"] == pytest.approx(0.70)
    assert result["leaf_count"] == len(_PEAKS)
    assert len(result["retrieval_id"]) == 32, "the join key back into both retrieval logs"
    # The rebuild input: what the template was bound with, never a second extraction.
    assert [c["name"] for c in result["slot_values"]["candidates"]] == _PEAKS
    assert result["slot_values"]["field"] == _FIELD
    # It is READ-ONLY: an action never mutates the graph.
    assert not node.children and node.details.get(DetailKey.ACTION_RESULT.value) is None

    # Every action result is sanitized onto its node; this one must survive that unchanged,
    # or the engine's rebuild would read back stringified slot values.
    assert sanitize_action_result(result) == result
    assert len(chroma.queries) == 1


@pytest.mark.asyncio
async def test_execute_reports_a_miss_without_spending_a_fill(monkeypatch, tmp_path):
    engine = _make_engine()
    _wire(engine, monkeypatch, tmp_path, [("argmax_t", 0.99, "argmax")])  # similarity 0.01
    graph = _graph()
    node = _search_node(graph)

    action = LeafActionRegistry(settings=engine.settings).get("plan_library_search")
    result = await action.execute(graph, node.node_id, engine.io)

    assert result["success"] is True, "a search that found nothing still ran fine"
    assert result["adopted"] is False and result["adopted_template_id"] is None
    assert result["decision"] == R.DECISION_NO_MATCH
    assert result["slot_values"] is None and result["leaf_count"] == 0


@pytest.mark.asyncio
async def test_execute_logs_the_attempt_under_the_on_demand_call_site(monkeypatch, tmp_path):
    """The shared pipeline logs both streams for BOTH call sites — only the label differs."""
    contract_out = tmp_path / "contract_log.jsonl"
    retrieval_out = tmp_path / "plan_retrievals.jsonl"
    monkeypatch.setenv("IDEA_TEST_CONTRACT_LOG", "1")
    monkeypatch.setenv(retrieval_log.ENV_FLAG, "1")
    engine = _make_engine()
    _wire(engine, monkeypatch, tmp_path, [("argmax_t", 0.30, "argmax")])
    graph = _graph()
    node = _search_node(graph)

    action = LeafActionRegistry(settings=engine.settings).get("plan_library_search")
    await action.execute(graph, node.node_id, engine.io)

    search = [r for r in _rows(contract_out) if r["event"] == "plan_library_search"]
    assert len(search) == 1
    assert search[0]["mode"] == R.CALL_SITE_ON_DEMAND == "on_demand"
    assert search[0]["adopted"] is True and search[0]["adopted_template_id"] == "argmax_t"

    rows = _rows(retrieval_out)
    assert len(rows) == 1
    assert rows[0]["call_site"] == R.CALL_SITE_ON_DEMAND
    assert rows[0]["slot_fill"] == "filled" and rows[0]["applied_template_id"] == "argmax_t"


@pytest.mark.asyncio
async def test_execute_fails_soft_when_the_library_explodes(monkeypatch, tmp_path):
    class _Raising:
        async def retrieve(self, *args, **kwargs):
            raise RuntimeError("chroma is down")

    monkeypatch.setattr(R, "PlanLibrary", lambda *a, **k: _Raising())
    engine = _make_engine(io=DummyIO(FakeChroma()))
    graph = _graph()
    node = _search_node(graph)

    action = LeafActionRegistry(settings=engine.settings).get("plan_library_search")
    result = await action.execute(graph, node.node_id, engine.io)

    assert result["success"] is False and result["action"] == "plan_library_search"
    assert "chroma is down" in result["error"]
    assert not node.children


# --------------------------------------------------------------------------------------
# adopted=True — the engine (not the action) grows the plan
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_adopted_search_grows_the_template_and_step_descends(monkeypatch, tmp_path):
    """The end-to-end claim, proven against the REAL engine loop rather than the hook alone:
    one step completes the search and materializes the template; the next step's existing
    routing (the `_got_reexpanded` escape hatch) drives the new children."""
    engine = _make_engine()
    engine.actions.register(RecordingSearchAction)
    chroma = _wire(engine, monkeypatch, tmp_path, [("argmax_t", 0.30, "argmax")])
    graph = _graph()
    node = _search_node(graph)

    assert await engine.step(graph, node.node_id, 0) == node.node_id

    # the search leaf itself: completed, marked, and now a parent
    result = node.details[DetailKey.ACTION_RESULT.value]
    assert result["adopted"] is True and result["adopted_template_id"] == "argmax_t"
    assert node.details["_got_reexpanded"] is True
    assert node.details["_got_reexpand_count"] == 1
    assert node.status == IdeaNodeStatus.ACTIVE

    # ...whose children are the template's filled leaves, wired like any library-sourced ones
    # (each with the page visit `link_page_visits` follows it through into — same wiring the
    # automatic path gets, because both call sites converge on the one expansion entry point)
    children = _searches(graph, node)
    visits = _visits(graph, node)
    assert len(children) == len(visits) == len(_PEAKS)
    for visit, search in zip(visits, children):
        assert visit.details[DetailKey.REQUIRES_DATA.value]["source_node_id"] == search.node_id
        assert visit.details[adapter.PLAN_LIBRARY_ORIGIN] == adapter.ORIGIN_ACTION
    for child, peak in zip(children, _PEAKS):
        assert child.details[adapter.PLAN_LIBRARY_ORIGIN] == adapter.ORIGIN_ACTION
        assert child.details[adapter.PLAN_LIBRARY_TEMPLATE_ID] == "argmax_t"
        assert child.details[DetailKey.ACTION.value] == IdeaActionType.SEARCH.value
        assert child.details[DetailKey.QUERY.value] == f"{peak} {_FIELD}"
        assert peak in child.details[DetailKey.EXPECT.value]
        # the children-metadata threading of the ONE expansion path, not a second copy of it
        assert child.details[DetailKey.IS_LEAF.value] is True
        assert child.details[DetailKey.PARENT_GOAL.value]
        assert child.details["_got_reexpand_count"] == 1
    # the template's aggregation discipline lands on the search node, for its merge
    assert node.details[DetailKey.INTENT.value] == (
        f"Write out every candidate's {_FIELD} in full before naming the winner."
    )

    # the rebuild cost nothing: no second ranking query, no second slot extraction
    assert len(chroma.queries) == 1

    # and now the load-bearing part — the engine's OWN routing drives the new subtree. Each
    # leaf's page visit waits on that leaf's search (`requires_data`), so the engine schedules
    # the searches itself rather than firing the whole batch off at once.
    current = node.node_id
    for step_index in range(1, 3 * len(children)):
        if all(c.status is IdeaNodeStatus.DONE for c in children):
            break
        current = await engine.step(graph, current, step_index) or node.node_id
    assert RecordingSearchAction.executed == [c.node_id for c in children]
    assert all(c.status == IdeaNodeStatus.DONE for c in children)


@pytest.mark.asyncio
async def test_an_adopted_search_respects_the_reexpansion_budget(monkeypatch, tmp_path):
    """Same termination bounds as every other trigger: `_reexpand_guards_ok` is not bypassed
    just because a template was found."""
    engine = _make_engine()
    _wire(engine, monkeypatch, tmp_path, [("argmax_t", 0.30, "argmax")])
    graph = _graph()
    node = _search_node(graph)
    # this lineage has already spent its single re-expansion
    node.details["_got_reexpand_count"] = engine._cfg.got.reexpand_max_iterations

    await engine.step(graph, node.node_id, 0)

    assert node.details[DetailKey.ACTION_RESULT.value]["adopted"] is True
    assert not node.children, "the budget guard wins over the retrieved plan"
    assert "_got_reexpanded" not in node.details
    assert node.status == IdeaNodeStatus.DONE


@pytest.mark.asyncio
async def test_an_unrebuildable_result_is_silence_not_a_crash(monkeypatch, tmp_path):
    """The template named by the result is gone from the corpus (an operator edited it between
    the search and its completion): fail toward silence, exactly like every other library
    failure — the leaf just stays a normal completed leaf."""
    engine = _make_engine()
    _wire(engine, monkeypatch, tmp_path, [("argmax_t", 0.30, "argmax")])
    graph = _graph()
    node = _search_node(graph)
    node.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.PLAN_LIBRARY_SEARCH.value,
        "success": True,
        "adopted": True,
        "adopted_template_id": "vanished_t",
        "slot_values": {"candidates": [{"name": p, "key": p.lower()} for p in _PEAKS]},
    }

    assert await engine._maybe_plan_library_reexpand(graph, node.node_id, 0) is False
    assert not node.children
    assert adapter.PLAN_LIBRARY_PENDING not in node.details


# --------------------------------------------------------------------------------------
# adopted=False — an ordinary completed leaf
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_miss_finalizes_as_a_normal_done_leaf(monkeypatch, tmp_path):
    engine = _make_engine()
    _wire(engine, monkeypatch, tmp_path, [("argmax_t", 0.99, "argmax")])  # similarity 0.01
    graph = _graph()
    node = _search_node(graph)

    next_id = await engine.step(graph, node.node_id, 0)

    assert next_id == graph.root_id(), "a completed leaf hands control back to its parent"
    assert node.status == IdeaNodeStatus.DONE
    assert node.details[DetailKey.ACTION_RESULT.value]["adopted"] is False
    assert not node.children
    assert "_got_reexpanded" not in node.details
    assert adapter.PLAN_LIBRARY_PENDING not in node.details
