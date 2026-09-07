# DAG Aggregation Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide, with a preregistered $0 experiment on the 23 aggregation-shaped tasks, whether DAG v2 is repairable by configuration and whether the deterministic coverage executor beats the linear arm on the shape where the graph loses −0.461.

**Architecture:** No new engine. A recovered task-shape registry, a frozen corpus for the aggregation tasks, one new deterministic `MechanicalEvaluationPolicy` selectable through settings, five arm profiles that flip already-built default-off mechanisms, and two preregistered campaigns (the configuration ablation, then the 2×2 with `evidence_loop`). The outcome routes to DAG v3 or to closing the engine question.

**Tech Stack:** Python 3.12, pytest, `scripts/build_corpus.py`, `scripts/prereg.py`, `scripts/run_campaign.sh`, `scripts/compare_arms.py --shapes`, corpus replay (`SEARCH_PROVIDER=corpus`).

**Spec:** `docs/superpowers/specs/2026-09-07-ledger-and-dag-next-gen-design.md` Part C and Part D (D1, D2, D3 decision; D4 deferred to after D1).

## Global Constraints

- No LLM changes. Learned or mechanical components replace judgements; prompts stay as they are except where a flag already alters them.
- Every campaign is seeded (`LLM_SEED`), corpus-replayed (`LEDGER_MAX_LIVE_FALLBACKS=0`), preregistered before launch, and audited with `prereg.py audit` before any number is read.
- Arm comparisons are paired by (model, task) and run-complete; no ranking from partial runs.
- Do not report per-shape splits at n < 20.
- The shape registry is written BLIND: from task-module source only, before any result is consulted, following the rule in `docs/AGGREGATION_SHAPE_FINDING_2026-08-30.md` §3.
- Test invocation: `PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests/<file>`.
- Commit style: single lowercase line, no punctuation, no body, no trailer.
- Corpus top-up is the only paid step (Serper, ~$0.10–0.30); it needs an explicit user go and a `--max-searches` cap.

---

## File Structure

| File | Responsibility |
|---|---|
| `agent/app/testing/task_shapes.json` (create) | the blind shape registry: `{"<test_id>": "aggregation" \| "chain" \| "other"}` for all 59 suite ids |
| `scripts/task_shapes.py` (create) | validate/print the registry; expose `aggregation_ids()`; used by tests and the campaign env |
| `agent/tests/task_shapes_test.py` (create) | registry covers suite59 exactly; counts within the doc's bands |
| `agent/idea_test_results/corpus/aggregation23/documents.jsonl` (create, untracked) | frozen evidence universe for the 23 tasks |
| `agent/app/idea_policies/mechanical_evaluation.py` (create) | `MechanicalEvaluationPolicy`: novelty + coverage score, no LLM |
| `agent/app/idea_engine.py` (modify ~155) | select the policy from `settings["evaluation_policy"]` |
| `agent/app/idea_test_runner.py` (modify `_GOT_ARM_PROFILES` ~558) | five `agg_*` profiles |
| `agent/tests/idea_test_runner_got_flags_test.py` (modify) | pin the new profiles |
| `agent/idea_test_results/prereg/agg_ablation_*.json`, `agg_2x2.json` (create) | preregistrations |
| `docs/handoffs/AGGREGATION_ABLATION_PREREG_2026-09-XX.md` (create) | hypothesis, env blocks, analysis commands, decision rule for D3 |

---

### Task 1: Recover the blind task-shape registry

**Files:**
- Create: `agent/app/testing/task_shapes.json`
- Create: `scripts/task_shapes.py`
- Test: `agent/tests/task_shapes_test.py`

**Interfaces:**
- Consumes: `scripts/adaptive_ladder_run.py:TASK_SETS["suite59"]` (the 59 ids); each `agent/app/idea_tests/test_<id>_*.py`'s `get_task_statement()` and its grep-validator names (`keystone_argmax`, `coverage`, `keystone_latest`, …).
- Produces: `task_shapes.load() -> Dict[str, str]`, `task_shapes.aggregation_ids() -> List[str]`, `task_shapes.chain_ids() -> List[str]`; the JSON also carries `"_rule"` and `"_written"` keys documenting the blind protocol.

- [ ] **Step 1: Write the failing test**

Create `agent/tests/task_shapes_test.py`:

```python
"""The task-shape registry recovered from the blind classification the aggregation finding was
built on. It must cover suite59 exactly, and the shape counts must sit inside the bands that
document reports (23 aggregation / 33 chain / 3 other), or the recovered registry is not the
same classification."""
from __future__ import annotations

import importlib.util
import pathlib

_SPEC = importlib.util.spec_from_file_location(
    "task_shapes", pathlib.Path(__file__).resolve().parents[2] / "scripts" / "task_shapes.py")
ts = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ts)

_LADDER = importlib.util.spec_from_file_location(
    "adaptive_ladder_run", pathlib.Path(__file__).resolve().parents[2] / "scripts" / "adaptive_ladder_run.py")
ladder = importlib.util.module_from_spec(_LADDER)
_LADDER.loader.exec_module(ladder)


def test_registry_covers_suite59_exactly():
    shapes = ts.load()
    ids = {k for k in shapes if not k.startswith("_")}
    assert ids == set(ladder.TASK_SETS["suite59"])


def test_every_value_is_one_of_the_three_shapes():
    shapes = ts.load()
    assert {v for k, v in shapes.items() if not k.startswith("_")} <= {"aggregation", "chain", "other"}


def test_counts_match_the_finding_within_two():
    assert abs(len(ts.aggregation_ids()) - 23) <= 2
    assert abs(len(ts.chain_ids()) - 33) <= 2


def test_registry_records_the_rule_and_blindness():
    shapes = ts.load()
    assert "combining evidence across" in shapes["_rule"]
    assert shapes["_written"].startswith("blind")
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests/task_shapes_test.py -v`
Expected: FAIL, script missing.

- [ ] **Step 3: Write the loader script**

Create `scripts/task_shapes.py`:

```python
#!/usr/bin/env python3
"""The blind task-shape registry for suite59.

`docs/AGGREGATION_SHAPE_FINDING_2026-08-30.md` §3 classified all 59 task modules from source
only (no result data) into aggregation vs chain; the classification itself was never saved.
This module owns the recovered registry (`agent/app/testing/task_shapes.json`) and is the ONLY
reader. Rule, verbatim: a task is AGGREGATION when the correct answer requires combining
evidence across two or more independently retrievable entities/rows (fan-out, count, argmax,
AND-filter); CHAIN when each hop depends on the previous hop's answer; OTHER when neither
(single-fact, disambiguation without combination).

Usage::
    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/task_shapes.py [--ids aggregation|chain]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "agent" / "app" / "testing" / "task_shapes.json"


def load(path: Path = REGISTRY) -> Dict[str, str]:
    return json.loads(Path(path).read_text())


def _ids(shape: str) -> List[str]:
    return sorted(k for k, v in load().items() if not k.startswith("_") and v == shape)


def aggregation_ids() -> List[str]:
    return _ids("aggregation")


def chain_ids() -> List[str]:
    return _ids("chain")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", choices=["aggregation", "chain", "other"])
    args = parser.parse_args(argv)
    if args.ids:
        print(",".join(_ids(args.ids)))
    else:
        shapes = load()
        for shape in ("aggregation", "chain", "other"):
            print(f"{shape}: {len(_ids(shape))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Write the registry, blind**

Protocol (do this before running any result-reading command):
1. `PYTHONPATH=.:services:agent ./.venv/bin/python -c "import sys; sys.path.insert(0,'scripts'); from adaptive_ladder_run import TASK_SETS; print(','.join(TASK_SETS['suite59']))"` → the 59 ids.
2. For each id, read only the module's `get_task_statement()` text and its validator names. Do not open any result file or the aggregation doc's per-task tables.
3. Apply the rule from the docstring. Write `agent/app/testing/task_shapes.json`:

```json
{
  "_rule": "aggregation when the correct answer requires combining evidence across two or more independently retrievable entities/rows (fan-out, count, argmax, AND-filter); chain when each hop depends on the previous hop's answer; other when neither",
  "_written": "blind, from task-module source and validator names only, 2026-09-XX, before any result file was opened",
  "001": "…", "002": "…"
}
```
(one entry per suite59 id; values `aggregation` / `chain` / `other`).

4. Only now run the tests. If the counts fall outside 23±2 / 33±2, do NOT edit labels to fit; record the disagreement in the JSON under `"_note"` and in the handoff doc. The finding's numbers are then re-derived on this registry, not assumed.

- [ ] **Step 5: Run the tests**

Run: `PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests/task_shapes_test.py -v`
Expected: PASS (or a documented count disagreement per step 4).

- [ ] **Step 6: Commit**

```bash
git add agent/app/testing/task_shapes.json scripts/task_shapes.py agent/tests/task_shapes_test.py
git commit -m "recover the blind suite59 task shape registry as data with a single reader"
```

---

### Task 2: Freeze a corpus for the aggregation tasks

**Files:**
- Create (untracked): `agent/idea_test_results/corpus/aggregation23/documents.jsonl`
- Modify: `docs/handoffs/AGGREGATION_ABLATION_PREREG_2026-09-XX.md` (record document count and cost)

**Interfaces:**
- Consumes: `scripts/build_corpus.py --results-dir --out --tests --live --max-searches --queries-per-task --visits-per-query`; `scripts/task_shapes.py --ids aggregation`.
- Produces: a corpus directory loadable by `SEARCH_PROVIDER=corpus LEDGER_CORPUS_DIR=agent/idea_test_results/corpus/aggregation23`.

- [ ] **Step 1: Free harvest from stored cells**

```bash
IDS=$(PYTHONPATH=.:services:agent ./.venv/bin/python scripts/task_shapes.py --ids aggregation)
PYTHONPATH=.:services:agent ./.venv/bin/python scripts/build_corpus.py \
    --results-dir agent/idea_test_results --out agent/idea_test_results/corpus/aggregation23 --tests "$IDS"
wc -l agent/idea_test_results/corpus/aggregation23/documents.jsonl
```
Expected: a document count printed. Record it.

- [ ] **Step 2: Paid top-up, only with a user go**

Aggregation tasks need 5–7 distinct pages each; the free harvest is biased toward the pages the losing arm actually reached. Ask the user for authorization with the cap stated, then:

```bash
export SERPER_KEY="$(grep '^SERPER_KEY=' services/keys.env | cut -d= -f2- | tr -d '"')"
PYTHONPATH=.:services:agent ./.venv/bin/python scripts/build_corpus.py \
    --results-dir agent/idea_test_results --out agent/idea_test_results/corpus/aggregation23 \
    --tests "$IDS" --live --max-searches 120 --queries-per-task 5 --visits-per-query 5
```
Expected: ≤120 searches (~$0.12 at Serper's rate), document count grows. Record both numbers in the handoff doc.

- [ ] **Step 3: Verify replay is byte-identical and search-free**

```bash
scripts/run_campaign.sh agg_corpus_smoke <<'ENVEOF'
SEARCH_PROVIDER=corpus
LEDGER_CORPUS_DIR=agent/idea_test_results/corpus/aggregation23
LEDGER_MAX_LIVE_FALLBACKS=0
LLM_SEED=22222
LLM_PROVIDER=ollama
MODEL_API_URL=http://127.0.0.1:11435/v1
OPENAI_API_KEY=dummy
IDEA_TEST_VALIDATION_MODEL=qwen2.5:7b
IDEA_TEST_IDS=052
IDEA_TEST_EXECUTION_VARIANTS=sequential_react
IDEA_TEST_RUNS=2
IDEA_TEST_CONCURRENCY=1
IDEA_TEST_JSON_TELEMETRY=1
IDEA_TEST_MODELS=qwen2.5:7b
IDEA_TEST_RUN_ID=agg_corpus_smoke
ENVEOF
```
Then `PYTHONPATH=.:services:agent ./.venv/bin/python scripts/run_diff.py --run-id agg_corpus_smoke` (or compare the two reps' `final_deliverable` by hand): expected identical deliverables and `live_fallbacks == 0` in both cells. Archive the smoke cells under `smoke_archive/`.

- [ ] **Step 4: Commit the doc line only**

The corpus is data under an ignored directory; commit the handoff doc update in Task 5.

---

### Task 3: `MechanicalEvaluationPolicy`

**Files:**
- Create: `agent/app/idea_policies/mechanical_evaluation.py`
- Modify: `agent/app/idea_engine.py:155` (policy selection)
- Test: `agent/tests/mechanical_evaluation_test.py`

**Interfaces:**
- Consumes: `EvaluationPolicy` base (`agent/app/idea_policies/base.py:20`, abstract `async evaluate(graph, node_id) -> float`); `IdeaDag.get_node`, `IdeaDag.has_executed_action(action_type, details) -> Optional[str]` (`idea_dag.py:376`), `graph.evaluate(node_id, score)` (`idea_dag.py:155`); `candidate_coverage.evaluate_candidate_coverage(graph, mandate) -> CandidateCoverageResult(.missing)`; node details keys `action`, `query`, `url`, `title`, `goal`, `mandate` (root).
- Produces: `class MechanicalEvaluationPolicy(EvaluationPolicy)` with `evaluate(graph, node_id) -> float` and `evaluate_batch(graph, parent_id, candidate_ids) -> Dict[str, float]`; scores in [0, 1]; writes `node.details["evaluation"] = {"score", "raw_score": None, "capped": False, "rationale": "mechanical", "signals": {...}}`. Engine reads `settings.get("evaluation_policy", "llm")`; `"mechanical"` selects this class.

Score, deterministic, from the external review's E1 list:
```
+0.35  action is search/visit and its key was not executed before (has_executed_action is None)
+0.35  node title/query/url names a candidate still in coverage.missing
+0.15  action is visit with a concrete URL (starts with http)
+0.15  query/url tokens not seen in any executed sibling's tokens (novelty)
-0.50  duplicate action key (has_executed_action returns an id)
clamp to [0, 1]
```

- [ ] **Step 1: Write the failing tests**

```python
import asyncio

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.mechanical_evaluation import MechanicalEvaluationPolicy

MANDATE = ("Which of these lakes is deepest?\n1. Lake Baikal — Siberia\n2. Crater Lake — Oregon\n"
           "3. Lake Tahoe — California\n")


def _graph():
    g = IdeaDag()
    root = g.add_root("root", {"mandate": MANDATE})
    return g, root.node_id


def test_a_fresh_visit_naming_a_missing_candidate_scores_high():
    g, root = _graph()
    n = g.add_child(root, "Visit Crater Lake", {"action": "visit", "url": "https://en.wikipedia.org/wiki/Crater_Lake"})
    score = asyncio.run(MechanicalEvaluationPolicy().evaluate(g, n.node_id))
    assert score >= 0.8
    assert g.get_node(n.node_id).details["evaluation"]["rationale"] == "mechanical"


def test_a_duplicate_action_scores_low():
    g, root = _graph()
    first = g.add_child(root, "Visit Baikal", {"action": "visit", "url": "https://en.wikipedia.org/wiki/Lake_Baikal"})
    g.get_node(first.node_id).details["action_result"] = {"success": True}
    g.get_node(first.node_id).status = "done"
    dup = g.add_child(root, "Visit Baikal again", {"action": "visit", "url": "https://en.wikipedia.org/wiki/Lake_Baikal"})
    score = asyncio.run(MechanicalEvaluationPolicy().evaluate(g, dup.node_id))
    assert score <= 0.3


def test_batch_returns_a_score_per_candidate():
    g, root = _graph()
    a = g.add_child(root, "Visit Tahoe", {"action": "visit", "url": "https://en.wikipedia.org/wiki/Lake_Tahoe"})
    b = g.add_child(root, "Think", {"action": "think", "goal": "reflect"})
    scores = asyncio.run(MechanicalEvaluationPolicy().evaluate_batch(g, root, [a.node_id, b.node_id]))
    assert set(scores) == {a.node_id, b.node_id}
    assert scores[a.node_id] > scores[b.node_id]
```
(Adjust `add_root`/`add_child` names and the status enum to `IdeaDag`'s real API: read `agent/app/idea_dag.py:59-110` first and mirror how `agent/tests/idea_dag_comprehensive_test.py` builds graphs.)

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests/mechanical_evaluation_test.py -v`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `agent/app/idea_policies/mechanical_evaluation.py`:

```python
"""A candidate scorer with no LLM in it.

The LLM evaluation judge measures at/below chance (node-level AUC 0.426; "which sibling ran"
0.378) and yet gates selection, pruning and beam width by default. This policy replaces it with
the mechanical signals the external review listed: novelty of the action key, whether the node
targets a still-missing roster candidate, URL concreteness, token novelty against executed
siblings, and a duplicate penalty. Deterministic, free, and selectable with
``settings["evaluation_policy"] = "mechanical"``.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import DetailKey, EvaluationPolicy
from agent.app.idea_policies.candidate_coverage import evaluate_candidate_coverage

_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_ACTION_KEYS = ("action", "query", "url", "title", "goal")


def _tokens(*texts: Any) -> Set[str]:
    out: Set[str] = set()
    for t in texts:
        out.update(_TOKEN_RE.findall(str(t or "").lower()))
    return out


def _mandate(graph: IdeaDag) -> str:
    root = graph.get_node(graph.root_id) if getattr(graph, "root_id", None) else None
    if root is None:
        for node in graph.nodes.values() if hasattr(graph, "nodes") else []:
            if not node.parent_id and not node.parent_ids:
                root = node
                break
    return str((root.details if root else {}).get("mandate") or "")


class MechanicalEvaluationPolicy(EvaluationPolicy):
    def __init__(self, settings: Optional[Dict[str, Any]] = None, **_: Any) -> None:
        super().__init__(settings)

    def _score(self, graph: IdeaDag, node_id: str, executed_tokens: Set[str]) -> Dict[str, Any]:
        node = graph.get_node(node_id)
        if node is None:
            return {"score": 0.0, "signals": {}}
        details = node.details or {}
        action = str(details.get("action") or "").lower()
        target = str(details.get("url") or details.get("query") or "")
        signals: Dict[str, float] = {}
        score = 0.0
        is_retrieval = action in ("search", "visit")
        dup = graph.has_executed_action(action, details) if is_retrieval else None
        if is_retrieval and dup is None:
            score += 0.35; signals["fresh_action"] = 1.0
        if dup is not None:
            score -= 0.50; signals["duplicate"] = 1.0
        mandate = _mandate(graph)
        missing: List[str] = []
        if mandate:
            try:
                missing = list(evaluate_candidate_coverage(graph, mandate).missing)
            except Exception:  # noqa: BLE001 -- coverage is advisory here
                missing = []
        node_tokens = _tokens(details.get("title"), target, details.get("goal"))
        if any(_tokens(name) & node_tokens for name in missing):
            score += 0.35; signals["targets_missing_candidate"] = 1.0
        if action == "visit" and target.startswith("http"):
            score += 0.15; signals["concrete_url"] = 1.0
        if node_tokens and not (node_tokens & executed_tokens):
            score += 0.15; signals["novel_tokens"] = 1.0
        return {"score": max(0.0, min(1.0, score)), "signals": signals}

    def _executed_tokens(self, graph: IdeaDag) -> Set[str]:
        out: Set[str] = set()
        for node in (graph.nodes.values() if hasattr(graph, "nodes") else []):
            details = node.details or {}
            if details.get(DetailKey.ACTION_RESULT.value) is not None:
                out |= _tokens(details.get("title"), details.get("url"), details.get("query"))
        return out

    async def evaluate(self, graph: IdeaDag, node_id: str) -> float:
        result = self._score(graph, node_id, self._executed_tokens(graph))
        graph.evaluate(node_id, result["score"])
        node = graph.get_node(node_id)
        if node is not None:
            node.details[DetailKey.EVALUATION.value] = {
                "score": result["score"], "raw_score": None, "capped": False,
                "rationale": "mechanical", "signals": result["signals"],
            }
        return result["score"]

    async def evaluate_batch(self, graph: IdeaDag, parent_id: str,
                             candidate_ids: List[str]) -> Dict[str, float]:
        executed = self._executed_tokens(graph)
        out: Dict[str, float] = {}
        for cid in candidate_ids:
            result = self._score(graph, cid, executed)
            graph.evaluate(cid, result["score"])
            node = graph.get_node(cid)
            if node is not None:
                node.details[DetailKey.EVALUATION.value] = {
                    "score": result["score"], "raw_score": None, "capped": False,
                    "rationale": "mechanical", "signals": result["signals"],
                }
            out[cid] = result["score"]
        return out
```
(Confirm `graph.nodes`, `graph.root_id`, and `DetailKey.EVALUATION`/`ACTION_RESULT` names against `idea_dag.py` and `base.py`; use the accessor the DAG actually exposes for iterating nodes.)

In `agent/app/idea_engine.py` at the `self.evaluation = ...` line:

```python
        if evaluation is not None:
            self.evaluation = evaluation
        elif str(self.settings.get("evaluation_policy", "llm")).lower() == "mechanical":
            from agent.app.idea_policies.mechanical_evaluation import MechanicalEvaluationPolicy
            self.evaluation = MechanicalEvaluationPolicy(settings=self.settings)
        else:
            self.evaluation = LlmBatchEvaluationPolicy(io=io, settings=self.settings, model_name=model_name)
```

- [ ] **Step 4: Run the tests and the engine suite**

Run: `PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests/mechanical_evaluation_test.py agent/tests/idea_engine_test.py -v` (if the engine test file has a different name, `ls agent/tests | grep idea_engine`).
Expected: PASS; the default (`"llm"`) path is byte-identical.

- [ ] **Step 5: Commit**

```bash
git add agent/app/idea_policies/mechanical_evaluation.py agent/app/idea_engine.py agent/tests/mechanical_evaluation_test.py
git commit -m "add a mechanical candidate scorer selectable by settings so the anti calibrated llm judge can be ablated"
```

---

### Task 4: The five `agg_*` arm profiles

**Files:**
- Modify: `agent/app/idea_test_runner.py:_GOT_ARM_PROFILES` (after `"good_adaptive_breadth"`)
- Modify: `agent/tests/idea_test_runner_got_flags_test.py`

**Interfaces:**
- Consumes: settings keys verified to exist: `got_prune_enabled`, `got_dynamic_beam_enabled`, `best_first_global`, `run_policy_sibling_evidence_digest_enabled`, `coverage_visit_injection_enabled`, `got_candidate_coverage_enabled`, `run_policy_evidence_store_mode`, `run_policy_deterministic_merge_view`, `run_policy_merge_uses_evidence_view`, `breadth_aware_branching_enabled`, plus Task 3's `evaluation_policy`.
- Produces: profiles `agg_default`, `agg_judges_off`, `agg_sibling_digest`, `agg_coverage_inject`, `agg_evidence_view`, `agg_all`, selectable with `IDEA_TEST_ARM=<name>`.

- [ ] **Step 1: Write the failing test**

Append to `agent/tests/idea_test_runner_got_flags_test.py`:

```python
def test_agg_profiles_exist_and_each_flips_exactly_its_own_lever():
    base = _GOT_ARM_PROFILES["agg_default"]
    assert _GOT_ARM_PROFILES["agg_judges_off"]["evaluation_policy"] == "mechanical"
    assert _GOT_ARM_PROFILES["agg_judges_off"]["got_prune_enabled"] is False
    assert _GOT_ARM_PROFILES["agg_judges_off"]["got_dynamic_beam_enabled"] is False
    assert _GOT_ARM_PROFILES["agg_sibling_digest"]["run_policy_sibling_evidence_digest_enabled"] is True
    assert _GOT_ARM_PROFILES["agg_coverage_inject"]["coverage_visit_injection_enabled"] is True
    assert _GOT_ARM_PROFILES["agg_coverage_inject"]["got_candidate_coverage_enabled"] is True
    assert _GOT_ARM_PROFILES["agg_evidence_view"]["run_policy_merge_uses_evidence_view"] is True
    assert _GOT_ARM_PROFILES["agg_evidence_view"]["run_policy_evidence_store_mode"] == "observe"
    combined = _GOT_ARM_PROFILES["agg_all"]
    for name in ("agg_judges_off", "agg_sibling_digest", "agg_coverage_inject", "agg_evidence_view"):
        for key, value in _GOT_ARM_PROFILES[name].items():
            assert combined[key] == value, (name, key)
    for name in ("agg_judges_off", "agg_sibling_digest", "agg_coverage_inject", "agg_evidence_view"):
        for key, value in base.items():
            assert _GOT_ARM_PROFILES[name].get(key, value) == value or key in _GOT_ARM_PROFILES[name]
```

- [ ] **Step 2: Run to verify failure** → `KeyError: 'agg_default'`.

- [ ] **Step 3: Add the profiles**

```python
    # Aggregation-shape ablation (docs/superpowers/specs/2026-09-07-ledger-and-dag-next-gen-design.md
    # D1). `agg_default` pins the shipped configuration explicitly so the four single-lever arms
    # differ from it by exactly one mechanism; `agg_all` is their union. Everything flipped here
    # already exists and ships default-off; nothing here touches a prompt.
    "agg_default": {
        "final_require_grounding": True,
        "breadth_aware_branching_enabled": True,     # the root can fan out to the roster
        "evaluation_policy": "llm",
        "got_prune_enabled": True,
        "got_dynamic_beam_enabled": True,
        "run_policy_sibling_evidence_digest_enabled": False,
        "coverage_visit_injection_enabled": False,
        "got_candidate_coverage_enabled": False,
        "run_policy_evidence_store_mode": "off",
        "run_policy_deterministic_merge_view": False,
        "run_policy_merge_uses_evidence_view": False,
    },
    "agg_judges_off": {
        **{}, "final_require_grounding": True, "breadth_aware_branching_enabled": True,
        "evaluation_policy": "mechanical", "got_prune_enabled": False, "got_dynamic_beam_enabled": False,
        "run_policy_sibling_evidence_digest_enabled": False, "coverage_visit_injection_enabled": False,
        "got_candidate_coverage_enabled": False, "run_policy_evidence_store_mode": "off",
        "run_policy_deterministic_merge_view": False, "run_policy_merge_uses_evidence_view": False,
    },
    "agg_sibling_digest": {
        "final_require_grounding": True, "breadth_aware_branching_enabled": True,
        "evaluation_policy": "llm", "got_prune_enabled": True, "got_dynamic_beam_enabled": True,
        "run_policy_sibling_evidence_digest_enabled": True, "coverage_visit_injection_enabled": False,
        "got_candidate_coverage_enabled": False, "run_policy_evidence_store_mode": "off",
        "run_policy_deterministic_merge_view": False, "run_policy_merge_uses_evidence_view": False,
    },
    "agg_coverage_inject": {
        "final_require_grounding": True, "breadth_aware_branching_enabled": True,
        "evaluation_policy": "llm", "got_prune_enabled": True, "got_dynamic_beam_enabled": True,
        "run_policy_sibling_evidence_digest_enabled": False, "coverage_visit_injection_enabled": True,
        "got_candidate_coverage_enabled": True, "run_policy_evidence_store_mode": "off",
        "run_policy_deterministic_merge_view": False, "run_policy_merge_uses_evidence_view": False,
    },
    "agg_evidence_view": {
        "final_require_grounding": True, "breadth_aware_branching_enabled": True,
        "evaluation_policy": "llm", "got_prune_enabled": True, "got_dynamic_beam_enabled": True,
        "run_policy_sibling_evidence_digest_enabled": False, "coverage_visit_injection_enabled": False,
        "got_candidate_coverage_enabled": False, "run_policy_evidence_store_mode": "observe",
        "run_policy_deterministic_merge_view": True, "run_policy_merge_uses_evidence_view": True,
    },
    "agg_all": {
        "final_require_grounding": True, "breadth_aware_branching_enabled": True,
        "evaluation_policy": "mechanical", "got_prune_enabled": False, "got_dynamic_beam_enabled": False,
        "run_policy_sibling_evidence_digest_enabled": True, "coverage_visit_injection_enabled": True,
        "got_candidate_coverage_enabled": True, "run_policy_evidence_store_mode": "observe",
        "run_policy_deterministic_merge_view": True, "run_policy_merge_uses_evidence_view": True,
    },
```
(Remove the stray `**{},` in `agg_judges_off`; it is shown only to flag that every profile is a full literal, not a `.update()` of another, so the pinning test can compare them key by key.)

- [ ] **Step 4: Run the flags test file** → PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/app/idea_test_runner.py agent/tests/idea_test_runner_got_flags_test.py
git commit -m "add the six aggregation ablation arm profiles each flipping one already built lever"
```

---

### Task 5: Preregister and document the two campaigns

**Files:**
- Create: `agent/idea_test_results/prereg/agg_ablation.json`, `agent/idea_test_results/prereg/agg_2x2.json`
- Create: `docs/handoffs/AGGREGATION_ABLATION_PREREG_2026-09-XX.md`

**Interfaces:**
- Consumes: `scripts/prereg.py write --spec`; `scripts/run_campaign.sh`; `scripts/compare_arms.py <arms> --shapes agent/app/testing/task_shapes.json --run-id <id>`; `scripts/task_shapes.py --ids aggregation`.
- Produces: two manifests, one handoff doc with the decision rule for D3. **Launch requires a user go** (GPU wall clock: ablation ≈ 23 tasks × 6 arms × 2 reps = 276 graph cells at ~3–5 min each ≈ 14–23 h; 2×2 ≈ 23 × 3 arms × 2 reps = 138 cells ≈ 4–6 h).

- [ ] **Step 1: Ablation prereg**

`/tmp/agg_ablation.json`:
```json
{
  "run_id": "agg_ablation",
  "hypothesis": "On the 23 aggregation-shaped suite59 tasks, the graph arm's paired deficit vs sequential_react (-0.461 at t=-7.73 on the original classification) is reduced by at least half by one of four already-built, default-off mechanisms (judges off, sibling digest, coverage injection, evidence view) or by their union.",
  "tasks": ["<the 23 ids from scripts/task_shapes.py --ids aggregation>"],
  "arms": ["agg_default", "agg_judges_off", "agg_sibling_digest", "agg_coverage_inject", "agg_evidence_view", "agg_all", "sequential_react"],
  "reps": 2,
  "primary_endpoint": "Paired (task, rep) mean-score delta of each agg_* arm vs sequential_react via scripts/compare_arms.py with --shapes; DECISION: if agg_all's deficit vs sequential_react is <= half of agg_default's deficit (both measured here, same corpus), DAG v2 is 'repairable by configuration' and D3-graph proceeds; if no single lever or the union halves the deficit, configuration repair is REJECTED. Secondary: which single lever moves it most (reported, not decision-bearing). Cost/wall-clock per arm reported alongside.",
  "abort_conditions": {"min_completion_rate": 0.95, "max_infra_failed_rate": 0.1, "max_live_fallbacks": 0}
}
```
The graph arm is selected with `IDEA_TEST_EXECUTION_VARIANTS=graph` and `IDEA_TEST_ARM=<profile>`; `sequential_react` runs with its own variant name and no arm profile. One `run_campaign.sh` launch per arm, serialized by the lockfile, all with `IDEA_TEST_RUN_ID=agg_ablation` so `compare_arms.py --run-id agg_ablation --tag <arm>` can pair them (confirm the runner writes the profile name into `run_config` or the file name; if not, use a distinct `IDEA_TEST_RUN_ID=agg_ablation_<arm>` per arm and pass each to `--arm`).

- [ ] **Step 2: 2×2 prereg**

`/tmp/agg_2x2.json`:
```json
{
  "run_id": "agg_2x2",
  "hypothesis": "On the same 23 aggregation tasks, the deterministic coverage executor (evidence_loop: (entity, field) rows + typed extraction + table-first finalize) scores at least +0.20 above sequential_react (paired), while sequential_react_extract (typed extraction only) isolates how much of that comes from representation vs scheduling.",
  "tasks": ["<the 23 ids>"],
  "arms": ["sequential_react", "sequential_react_extract", "evidence_loop"],
  "reps": 2,
  "primary_endpoint": "Paired delta evidence_loop - sequential_react on mean score via compare_arms.py; DECISION: delta >= +0.20 with p < 0.05 -> D3 'coverage DAG' is justified; delta in (-0.10, +0.20) -> inconclusive, report and stop engine work; delta <= -0.10 -> linear wins, engine question closed. Secondary: extract-only arm's delta (representation share), tokens and wall-clock per arm, distinct pages per cell.",
  "abort_conditions": {"min_completion_rate": 0.95, "max_infra_failed_rate": 0.1, "max_live_fallbacks": 0}
}
```

- [ ] **Step 3: Write both manifests**

```bash
for s in agg_ablation agg_2x2; do PYTHONPATH=.:services:agent ./.venv/bin/python scripts/prereg.py write --spec /tmp/$s.json; done
```
(Replace the placeholder task lists with the real ids first.)

- [ ] **Step 4: Write the handoff doc**

`docs/handoffs/AGGREGATION_ABLATION_PREREG_2026-09-XX.md` sections: (1) why (Part C of the spec, three sentences); (2) the registry recovery note from Task 1 step 4 including any count disagreement; (3) corpus provenance (document counts, searches, cost) from Task 2; (4) both hypotheses verbatim; (5) launch blocks: one `run_campaign.sh` heredoc per arm, copying Task 2 step 3's env with `IDEA_TEST_IDS=$(scripts/task_shapes.py --ids aggregation)` expanded, `IDEA_TEST_RUNS=2`, `IDEA_TEST_EXECUTION_VARIANTS=graph` + `IDEA_TEST_ARM=<profile>` for graph arms; (6) analysis commands: `prereg.py audit --run-id ...` first, then `compare_arms.py`; (7) the D3 decision table copied from the two prereg endpoints; (8) "Launch requires a user go; projected wall clock; $0 inference."

- [ ] **Step 5: Commit**

```bash
git add agent/idea_test_results/prereg/agg_ablation.json agent/idea_test_results/prereg/agg_2x2.json docs/handoffs/AGGREGATION_ABLATION_PREREG_2026-09-XX.md
git commit -m "preregister the aggregation ablation and the coverage executor 2x2 with their decision rules"
```

---

### Task 6 (after the campaigns, not before): analysis and the D3 decision record

**Files:**
- Create: `docs/handoffs/AGGREGATION_ABLATION_RESULTS_2026-09-XX.md`
- Modify: `docs/STATE_OF_EVIDENCE_2026-09-07.md` (add the result rows), `docs/handoffs/ROADMAP_2026-09-07.md` (record the D3 branch taken)

- [ ] **Step 1: Audit before reading numbers**

```bash
PYTHONPATH=.:services:agent ./.venv/bin/python scripts/prereg.py audit --run-id agg_ablation
PYTHONPATH=.:services:agent ./.venv/bin/python scripts/prereg.py audit --run-id agg_2x2
```
Expected: every gate PASS; otherwise stop and report which gate failed. Grep every cell log for `Setup failed` before trusting a local run.

- [ ] **Step 2: Compare**

```bash
PYTHONPATH=.:services:agent ./.venv/bin/python scripts/compare_arms.py agg_default agg_judges_off agg_sibling_digest agg_coverage_inject agg_evidence_view agg_all sequential_react \
    --run-id agg_ablation --shapes agent/app/testing/task_shapes.json --rep 2
PYTHONPATH=.:services:agent ./.venv/bin/python scripts/compare_arms.py sequential_react sequential_react_extract evidence_loop \
    --run-id agg_2x2 --shapes agent/app/testing/task_shapes.json --rep 2
```

- [ ] **Step 3: Write the results doc** with: the prereg decision rules quoted, the measured deltas with t and n, the branch taken (D3-graph / stop engine work / linear wins), cost and wall-clock per arm, and a "may / may not be claimed" list. Update the state-of-evidence table and the roadmap.

- [ ] **Step 4: Commit**

```bash
git add docs/handoffs/AGGREGATION_ABLATION_RESULTS_2026-09-XX.md docs/STATE_OF_EVIDENCE_2026-09-07.md docs/handoffs/ROADMAP_2026-09-07.md
git commit -m "record the aggregation ablation and 2x2 results and the dag v3 decision they force"
```

---

## Deferred, deliberately

- **D4 learned stop/continue** waits for the ablation result: if judges-off wins, the stopping rule's inputs change, and fitting it now would be fitting to the wrong arm.
- **D3 build** (rows as work items, dependency-aware queue, parallel dispatch) is its own spec and plan, written only if the 2×2 justifies it.

## Self-review against the spec

- C1/C4 → Tasks 3, 4 (judges off; four built-but-off mechanisms). C5/D2 → Task 5's 2×2. D1 → Tasks 1, 2, 4, 5. D3 → Task 6's decision. D4 → deferred, stated.
- Placeholders: the registry values in Task 1 and the task-id lists in Task 5 are intentionally filled at execution time because the blind protocol forbids pre-listing them here; both steps say exactly how to produce them.
- Type consistency: `evaluation_policy` string key is read in `idea_engine.py` and set in the profiles; `MechanicalEvaluationPolicy.evaluate_batch` matches the engine's `hasattr(self.evaluation, "evaluate_batch")` call site; `task_shapes.aggregation_ids()` is what Task 2 and Task 5 consume.
