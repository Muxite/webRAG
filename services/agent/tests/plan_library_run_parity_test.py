"""Full-run parity: a plan library that never fires must change NOTHING.

``plan_library_auto_shortcircuit_test`` proves non-dependency at ONE node — flag off, no
``PlanLibrary`` is built and ``_handle_expansion_node`` behaves exactly as before. That is not
the claim the design actually rests on. ``RESEARCH_LIBRARY.md``'s ``plan_library`` entry states
it run-wide: "a missing/empty/broken index, a below-threshold similarity, or a failed slot-fill
all degrade silently to organic expansion, never to a wrong or partial plan" — i.e. an ARMED
library that misses must be indistinguishable, over a whole run, from no library at all. A
single-node check cannot see a divergence that only shows up several steps later (a different
step budget, an extra grounding replan, a different finalize signal).

So this file runs the same scripted scenario twice through the SAME entry point
``control_loop_parity_test`` uses — ``IdeaDagEngine.run()`` — differing only in the two
plan-library flags, and compares that file's own ``PARITY_KEYS`` finalize signals (imported, so
the two parity suites can never drift apart on what "identical output" means) plus the whole
graph's shape and the expansion-policy call count. The scenario deliberately consults the
library twice (root expansion, then ``_run_loop``'s emergency re-expansion) and then runs on
through action execution, the merge, two grounding replans and the candidate-coverage gate, so
a divergence introduced anywhere downstream of the miss is visible.

Two independent ways of missing are covered, mirroring
``plan_library_auto_shortcircuit_test``'s own miss mechanisms: a real corpus whose best match
is far below the calibrated threshold (``weak_match``), and an EMPTY corpus with an index that
returns nothing (``empty_corpus``). The anti-no-op guard matters as much as the parity
assertion here — a test where retrieval silently never ran would pass for the wrong reason — so
the armed run also asserts that the library really was built, really was queried, and really
returned ``no_match`` every time. (Verified to have teeth: pointing the same fixture at a
CONFIDENT match breaks both the graph-shape and the ``got_stats`` assertions.)

Fully offline: fake expansion/evaluation/selection/merge policies, a scripted action registry,
a ``ConnectorChroma``-shaped fake and a mocked ``build_final_payload``. No LLM/search/http call
is ever made.
"""
from __future__ import annotations

import json

import pytest

import agent.app.idea_engine as engine_mod
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies import BestScoreSelectionPolicy, SimpleMergePolicy
from agent.app.idea_policies import plan_library as adapter
from agent.app.idea_policies.base import (
    DetailKey,
    DecompositionPolicy,
    EvaluationPolicy,
    ExpansionPolicy,
    IdeaActionType,
)
from agent.app.plan_library import retrieval as R

# The finalize-signal contract is defined ONCE, by the control-loop parity suite; importing it
# means a signal added there is automatically covered here too.
from agent.tests.control_loop_parity_test import BASE_MANDATE, PARITY_KEYS

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
    ],
    "leaves": [
        {
            "id_pattern": "<<item.key>>_field",
            "for_each": "candidates",
            "instruction": "Open the authoritative page for <<item.name>> and read it.",
            "expect": "<<item.name>>: the value read -- source URL",
        },
    ],
    "aggregation": "Write out every candidate in full before naming the winner.",
}

# What the LLM invents when the library does not answer, scripted per expansion call. The
# first call is deliberately EMPTY: the root then has no children after step 1, which is the
# `_run_loop` emergency-re-expansion branch — so the library is consulted (and misses) more
# than once per run, and the miss is compared across a loop branch a node-local test cannot
# reach at all. The second call is a real plan: a search whose result grounds a page read.
_ORGANIC_CANDIDATES = [
    [],
    [
        {
            "title": "Search for which River Avon empties into the English Channel",
            "details": {
                DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
                DetailKey.QUERY.value: "River Avon Hampshire mouth English Channel",
                DetailKey.EXPECT.value: "the Avon reaching the English Channel -- source URL",
                DetailKey.IS_LEAF.value: True,
            },
        },
        {
            "title": "Read the River Avon, Hampshire page",
            "details": {
                DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                DetailKey.URL.value: "https://en.wikipedia.org/wiki/River_Avon,_Hampshire",
                DetailKey.EXPECT.value: "River Avon, Hampshire: its mouth -- source URL",
                DetailKey.IS_LEAF.value: True,
            },
        },
    ],
]

_SEARCH_RESULT = {
    "action": IdeaActionType.SEARCH.value,
    "success": True,
    "results": [
        {
            "title": "River Avon, Hampshire",
            "url": "https://en.wikipedia.org/wiki/River_Avon,_Hampshire",
        }
    ],
}
_VISIT_RESULT = {
    "action": IdeaActionType.VISIT.value,
    "success": True,
    "url": "https://en.wikipedia.org/wiki/River_Avon,_Hampshire",
    "content": (
        "The River Avon, Hampshire flows into the English Channel at Christchurch Harbour."
    ),
    "links": [],
}


# --------------------------------------------------------------------------------------
# fakes (the ``plan_library_auto_shortcircuit_test`` set, plus a scripted action registry)
# --------------------------------------------------------------------------------------


class _FakeCollection:
    def __init__(self, space="cosine"):
        self.configuration_json = {"hnsw": {"space": space}}
        self.metadata = {"hnsw:space": space}


class FakeChroma:
    """The ``ConnectorChroma`` surface retrieval uses; every other method is a no-op.

    Both runs get one of these, so the memory manager's own Chroma traffic is identical on
    each side and the ONLY difference between them is the two plan-library flags.
    """

    def __init__(self, hits=()):
        self.hits = list(hits)  # [(template_id, distance, archetype), ...]
        self.queries = []

    async def get_or_create_collection(self, collection, metadata=None):
        return _FakeCollection()

    async def query_chroma(self, collection, query_texts, n_results=3, where=None):
        self.queries.append((collection, list(query_texts)))
        hits = self.hits[:n_results]
        return {
            "ids": [[h[0] for h in hits]],
            "distances": [[h[1] for h in hits]],
            "metadatas": [[{"template_id": h[0], "archetype": h[2]} for h in hits]],
        }


class DummyIO:
    """An ``AgentIO`` stand-in with a vector DB but deliberately no LLM."""

    def __init__(self, connector_chroma=None):
        self.connector_chroma = connector_chroma

    def set_telemetry(self, telemetry):
        return None


class FakeExpansion(ExpansionPolicy):
    """The LLM-invented path, scripted per call. Its call count is half of what parity means
    here: the same plan must be invented the same number of times on both sides."""

    def __init__(self, settings=None):
        super().__init__(settings=settings)
        self.calls = 0

    async def expand(self, graph, node_id, memories=None):
        script = _ORGANIC_CANDIDATES[min(self.calls, len(_ORGANIC_CANDIDATES) - 1)]
        self.calls += 1
        return [json.loads(json.dumps(c)) for c in script]


class FakeEvaluation(EvaluationPolicy):
    async def evaluate(self, graph, node_id):
        graph.evaluate(node_id, 0.6)
        return 0.6

    async def evaluate_batch(self, graph, parent_id, candidate_ids):
        return {nid: (graph.evaluate(nid, 0.6) or 0.6) for nid in candidate_ids}


class FakeDecomposition(DecompositionPolicy):
    def should_decompose(self, graph, node_id):
        return False


class _ScriptedAction:
    def __init__(self, result, provides=None):
        self._result = result
        self._provides = provides

    async def execute(self, graph, node_id, io):
        return json.loads(json.dumps(self._result))

    def post_execute_provides(self, node, result):
        """Mirrors the real actions' contract tagging — a search's ``urls_from_search`` is
        what makes a downstream ``requires_data`` visit executable at all."""
        return self._provides


class ScriptedActionRegistry:
    """Deterministic, offline stand-ins for the real connector-backed actions."""

    def __init__(self, settings):
        self.settings = dict(settings or {})
        self._by_type = {
            action_type: _ScriptedAction({"action": action_type.value, "success": True})
            for action_type in IdeaActionType
        }
        self._by_type[IdeaActionType.SEARCH] = _ScriptedAction(
            _SEARCH_RESULT, provides="urls_from_search"
        )
        self._by_type[IdeaActionType.VISIT] = _ScriptedAction(_VISIT_RESULT)

    def get(self, action_type):
        return self._by_type.get(action_type)


class _CountingCorpus:
    """Wraps the real ``PlanLibrary`` to record every retrieval decision (anti-no-op guard)."""

    def __init__(self, inner):
        self._inner = inner
        self.decisions = []

    async def retrieve(self, *args, **kwargs):
        result = await self._inner.retrieve(*args, **kwargs)
        self.decisions.append(getattr(result, "decision", None))
        return result

    async def fill_from_query(self, *args, **kwargs):  # pragma: no cover - a miss never fills
        return await self._inner.fill_from_query(*args, **kwargs)

    def get(self, template_id):
        return self._inner.get(template_id)


def _settings(*, plan_library_on: bool):
    """Identical on both sides except the two flags under test."""
    return {
        "plan_library_enabled": plan_library_on,
        "plan_library_auto_enabled": plan_library_on,
        "allow_unscored_selection": True,
        "min_score_threshold": 0.0,
        "best_first_global": False,
        "got_dedup_enabled": False,
        "got_embed_on_create": False,
        "auto_parallel_siblings": False,
        "semantic_dedup_visits_enabled": False,
        "got_candidate_coverage_enabled": True,
        "got_backtrack_enabled": True,
        "got_prune_interval_steps": 2,
    }


def _make_engine(*, plan_library_on: bool, hits):
    settings = _settings(plan_library_on=plan_library_on)
    expansion = FakeExpansion(settings)
    chroma = FakeChroma(hits)
    engine = IdeaDagEngine(
        io=DummyIO(chroma),
        settings=settings,
        model_name="m",
        expansion=expansion,
        evaluation=FakeEvaluation(settings),
        selection=BestScoreSelectionPolicy(settings=settings),
        decomposition=FakeDecomposition(settings),
        merge=SimpleMergePolicy(settings=settings),
        actions=ScriptedActionRegistry(settings),
        post_expansion_hooks=[],
    )
    return engine, expansion, chroma


def _library(tmp_path, templates) -> R.PlanLibrary:
    tmp_path.mkdir(parents=True, exist_ok=True)
    for template in templates:
        (tmp_path / f"{template['template_id']}.json").write_text(
            json.dumps(template), encoding="utf-8"
        )
    return R.PlanLibrary(templates_dir=tmp_path, warn_on_drift=False)


#: The two independent ways an armed library can come up empty, both already exercised
#: node-locally by ``plan_library_auto_shortcircuit_test``. The third field says whether the
#: miss is reached THROUGH the index — an empty corpus has nothing to rank, so retrieval
#: reports ``no_match`` without spending a query.
_MISS_MODES = {
    # A real corpus, ranked — but the best match's similarity (1 - 0.99) is far below the
    # calibrated auto-apply threshold, so retrieval reports ``no_match``.
    "weak_match": ((_ARGMAX_TEMPLATE,), [("argmax_t", 0.99, "argmax")], True),
    # Nothing to match against at all: the "missing/empty index" case.
    "empty_corpus": ((), [], False),
}


async def _run(tmp_path, *, plan_library_on: bool, miss_mode: str, max_steps: int = 8):
    templates, hits, _ = _MISS_MODES[miss_mode]
    engine, expansion, chroma = _make_engine(plan_library_on=plan_library_on, hits=hits)
    corpus = None
    if plan_library_on:
        # The lazy-cache seam: a temp corpus instead of the shipped one, wrapped so every
        # retrieval decision is recorded.
        corpus = _CountingCorpus(_library(tmp_path, templates))
        engine._plan_library_corpus_cache = corpus
    payload = await engine.run(BASE_MANDATE, max_steps=max_steps)
    return payload, engine, expansion, chroma, corpus


def _shape(graph_dict, node_id=None):
    """The graph's structure, free of the uuids that legitimately differ between two runs."""
    node_id = node_id or graph_dict["root_id"]
    node = graph_dict["nodes"][node_id]
    details = node.get("details") or {}
    return (
        node.get("title"),
        node.get("status"),
        details.get(DetailKey.ACTION.value),
        details.get(DetailKey.INTENT.value),
        tuple(_shape(graph_dict, cid) for cid in node.get("children") or []),
    )


@pytest.fixture(autouse=True)
def _fixed_finalize(monkeypatch):
    """No LLM in finalize; ``finalize()`` layers the parity signals on top of this."""

    async def _fixed_payload(*args, **kwargs):
        return {"final_deliverable": "answer", "success": True}

    monkeypatch.setattr(engine_mod, "build_final_payload", _fixed_payload)


@pytest.mark.parametrize("miss_mode", sorted(_MISS_MODES))
@pytest.mark.asyncio
async def test_a_library_that_never_fires_changes_nothing_about_the_run(tmp_path, miss_mode):
    """The load-bearing guarantee, run-wide: same finalize signals, same graph, same work."""
    armed, armed_engine, armed_expansion, armed_chroma, corpus = await _run(
        tmp_path / "armed", plan_library_on=True, miss_mode=miss_mode
    )
    off, off_engine, off_expansion, off_chroma, _ = await _run(
        tmp_path / "off", plan_library_on=False, miss_mode=miss_mode
    )

    for key in PARITY_KEYS:
        assert armed.get(key) == off.get(key), f"parity mismatch on {key!r}"
    # Beyond the finalize signals: the plan the engine actually built, node for node.
    assert _shape(armed["graph"]) == _shape(off["graph"])
    # ...and the same amount of LLM work to build it.
    assert armed_expansion.calls == off_expansion.calls

    # Anti-no-op guards. Without these the test would pass just as happily if the mechanism
    # had never run at all.
    assert armed_expansion.calls == len(_ORGANIC_CANDIDATES), "the whole script really ran"
    assert armed_engine._plan_library_corpus_cache is corpus, "the armed run kept its corpus"
    assert len(corpus.decisions) == armed_expansion.calls, "every expansion asked the library"
    assert set(corpus.decisions) == {R.DECISION_NO_MATCH}, "...and it missed every time"
    # The flag-off run never even builds a library (so it never reads the corpus off disk).
    assert off_engine._plan_library_corpus_cache is None
    assert not [c for c, _ in off_chroma.queries if c == R.COLLECTION_NAME]
    # ...and where the miss IS reached through the index, the armed run really queried it.
    _, _, misses_through_the_index = _MISS_MODES[miss_mode]
    if misses_through_the_index:
        assert [c for c, _ in armed_chroma.queries if c == R.COLLECTION_NAME]


@pytest.mark.parametrize("miss_mode", sorted(_MISS_MODES))
@pytest.mark.asyncio
async def test_a_missed_run_leaves_no_plan_library_marker_on_any_node(tmp_path, miss_mode):
    """A miss must not attribute anything: no origin, no template id, no pending expansion
    anywhere in the finished graph — the ``contract_made`` join key stays ``organic``."""
    armed, _, _, _, _ = await _run(tmp_path, plan_library_on=True, miss_mode=miss_mode)

    markers = (
        adapter.PLAN_LIBRARY_ORIGIN,
        adapter.PLAN_LIBRARY_TEMPLATE_ID,
        adapter.PLAN_LIBRARY_PENDING,
    )
    for node in armed["graph"]["nodes"].values():
        details = node.get("details") or {}
        assert not [m for m in markers if m in details], f"{node.get('title')!r} was attributed"
