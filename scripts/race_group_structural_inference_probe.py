"""Live ($0, local-ollama) validation probe for the STRUCTURAL race-group inference added in
commit d217e830 (``alternative_branch.infer_race_groups``).

The older ``scripts/alt_branch_emission_probe.py`` measured whether a model ever TYPES the
``race_group``/``alternative_of`` tag. That probe cannot validate this mechanism, which never
asks the model for a tag: it reads the race relationship off plan SHAPE after expansion. So
this probe drives the REAL expansion path end to end —

    LlmExpansionPolicy.expand()  (real prompt, real schema hint, real _parse_candidates)
        -> IdeaDag.expand()      (real nodes minted from the real completion)
        -> alternative_branch.infer_race_groups(parent, created_children)   (the real pass,
           same call the engine makes at idea_engine.py's post-expand wiring)

— and reports the resulting ``RACE_GROUPS_INFERRED`` registry state on the parent, not a
string match on raw text.

Three arms per (model, mandate), so the "structural fix changed X where prompt-only got zero"
claim is measured in ONE run rather than cited from an earlier probe:

* ``struct``        — ``expansion_race_group_structural_inference_enabled`` ON, branching
                      schema variant OFF, ``expect`` contract OFF. Tier 2 (title overlap)
                      is the only same-target signal available.
* ``struct+expect`` — as above plus ``expansion_expect_contract_enabled``, which is what makes
                      the stronger tier 1 (near-duplicate ``expect``) signal reachable.
* ``tag``           — the OLD mechanism: ``expansion_alternative_branch_enabled`` ON, counting
                      raw ``alternative_of``/``race_group`` emission AND whether
                      ``link_race_groups`` actually registered an authored group.

Five stimuli: the four positive shapes from the emission probe (the mechanism SHOULD fire) plus
task 052's REAL task statement as a breadth negative control (a six-way fan-out over six
unrelated entities — the mechanism must NOT fire; a hit there is a false positive).

Not a pytest test — a standalone diagnostic, run manually:
    PYTHONPATH=.:services:agent python3 scripts/race_group_structural_inference_probe.py

$0 cost: local ollama inference only, no OpenRouter/live-API calls.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from agent.app.idea_dag import IdeaDag  # noqa: E402
from agent.app.idea_policies import alternative_branch as _alt  # noqa: E402
from agent.app.idea_policies.expansion import LlmExpansionPolicy  # noqa: E402
from agent.app.idea_tests import test_052_tier5_breadth_aggregation as _t052  # noqa: E402
from agent.app.llm_backends import LLMBackend  # noqa: E402
from scripts.alt_branch_emission_probe import STIMULI as POSITIVE_STIMULI  # noqa: E402

#: The four positive shapes (150/122-derived, reused verbatim from the emission probe so the
#: before/after comparison is on identical stimuli) plus the breadth negative control, whose
#: mandate is task 052's REAL statement rather than a hand-written imitation.
STIMULI: List[Dict[str, str]] = list(POSITIVE_STIMULI) + [
    {
        "id": "052_breadth_negative_control",
        "mandate": _t052.get_task_statement(),
        "negative_control": True,
    },
]

MODELS = ["qwen2.5:0.5b", "qwen2.5:7b", "qwen2.5:14b"]

ARMS: Dict[str, Dict[str, Any]] = {
    "struct": {
        "expansion_race_group_structural_inference_enabled": True,
        "expansion_alternative_branch_enabled": False,
        "expansion_expect_contract_enabled": False,
    },
    "struct+expect": {
        "expansion_race_group_structural_inference_enabled": True,
        "expansion_alternative_branch_enabled": False,
        "expansion_expect_contract_enabled": True,
    },
    "tag": {
        "expansion_race_group_structural_inference_enabled": False,
        "expansion_alternative_branch_enabled": True,
        "expansion_expect_contract_enabled": False,
    },
}


def _make_connector():
    """The real ``ConnectorLLM`` with a stubbed backend: ``build_payload`` passes ``messages``
    through untouched, so the payload the policy hands us is the payload the engine builds."""
    from shared.connector_config import ConnectorConfig

    cfg = ConnectorConfig()
    mock_backend = AsyncMock(spec=LLMBackend)
    mock_backend.normalize_payload.side_effect = lambda p, *_, **__: p
    mock_backend.simplify_payload.side_effect = lambda p: dict(p)
    with patch("agent.app.connector_llm.create_llm_backend", return_value=mock_backend):
        from agent.app.connector_llm import ConnectorLLM

        return ConnectorLLM(cfg)


class _OllamaIO:
    """``AgentIO`` stand-in that routes the policy's own payload to local ollama."""

    def __init__(self, base_url: str, model: str, num_predict: int, timeout: int):
        self._connector = _make_connector()
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._num_predict = num_predict
        self._timeout = timeout
        self.completion: str = ""

    def set_telemetry(self, telemetry):  # pragma: no cover - interface shim
        return None

    def build_llm_payload(self, **kwargs):
        return self._connector.build_payload(**kwargs)

    async def query_llm_with_fallback(self, payload, **kwargs):
        messages = payload.get("messages") or []
        body: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": self._num_predict,
                "temperature": payload.get("temperature", 0.2),
            },
        }
        # The engine asks for json_object mode; ollama's equivalent is format=json.
        if payload.get("response_format"):
            body["format"] = "json"
        resp = await asyncio.to_thread(
            requests.post, f"{self._base_url}/api/chat", json=body, timeout=self._timeout
        )
        resp.raise_for_status()
        content = str(resp.json().get("message", {}).get("content", ""))
        self.completion = content
        return content


def _summarize_registry(parent, key: str) -> Optional[Dict[str, List[str]]]:
    value = parent.details.get(key)
    return value if isinstance(value, dict) and value else None


async def run_cell(model: str, stim: Dict[str, str], arm: str, args) -> Dict[str, Any]:
    """One (model, mandate, arm) cell: real expand -> real nodes -> real inference pass."""
    io = _OllamaIO(args.base_url, model, args.num_predict, args.timeout)
    policy = LlmExpansionPolicy(io=io, settings=dict(ARMS[arm]), model_name=model)
    graph = IdeaDag(root_title="Research", root_details={"mandate": stim["mandate"]})
    root_id = graph.root_id()
    parent = graph.get_node(root_id)

    try:
        candidates = await policy.expand(graph, root_id)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}

    children = graph.expand(root_id, candidates)
    cell: Dict[str, Any] = {
        "candidates": len(children),
        "titles": [c.title for c in children],
        # Per-node shape, because a tier-1 miss is only diagnosable with the pairing of
        # `expect` to action/url: an identical-`expect` cluster is still (correctly) rejected
        # when its members are not demonstrably disjoint approaches.
        "nodes": [
            {
                "title": c.title,
                "action": c.details.get("action"),
                "expect": c.details.get("expect"),
                "url": c.details.get("url") or c.details.get("optional_url"),
            }
            for c in children
        ],
        "completion_chars": len(io.completion),
    }

    lower = (io.completion or "").lower()
    cell["raw_alternative_of"] = lower.count("alternative_of")
    cell["raw_race_group"] = lower.count("race_group")

    if arm == "tag":
        # The real post-expand wiring the engine runs behind `alternative_branch_enabled`.
        cell["parked"] = _alt.link_alternatives(children)
        cell["registered"] = _alt.link_race_groups(parent, children)
        cell["race_groups"] = _summarize_registry(parent, _alt.RACE_GROUPS)
    else:
        # The real post-expand wiring the engine runs behind
        # `race_group_structural_inference_enabled` (idea_engine.py ~L1523).
        tiers = _alt.infer_race_groups(parent, children)
        cell["tiers"] = tiers
        cell["inferred"] = _summarize_registry(parent, _alt.RACE_GROUPS_INFERRED)
        cell["inferred_tiers"] = _summarize_registry(parent, _alt.RACE_GROUPS_INFERRED_TIERS)
        cell["expects"] = [
            c.details.get("expect") for c in children if c.details.get("expect")
        ]
    return cell


def _cell_label(arm: str, cell: Dict[str, Any]) -> str:
    if "error" in cell:
        return "ERR"
    if arm == "tag":
        groups = cell.get("race_groups") or {}
        raw = cell.get("raw_race_group", 0) + cell.get("raw_alternative_of", 0)
        if groups:
            return f"{len(groups)}grp(tag)"
        if cell.get("parked"):
            return f"{cell['parked']}alt"
        return "none" if not raw else f"none(raw={raw})"
    inferred = cell.get("inferred") or {}
    if not inferred:
        return "none"
    tiers = cell.get("inferred_tiers") or {}
    tier_str = ",".join(str(tiers.get(label, "?")) for label in inferred)
    return f"{len(inferred)}grp/T{tier_str}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:11435")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--num-predict", type=int, default=1200)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    try:
        requests.get(f"{args.base_url.rstrip('/')}/api/tags", timeout=5).raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"No local ollama instance reachable at {args.base_url}: {exc}")
        return 1

    models = [m for m in args.models.split(",") if m.strip()]
    arms = [a for a in args.arms.split(",") if a.strip()]
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for model in models:
        for stim in STIMULI:
            for arm in arms:
                cell = asyncio.run(run_cell(model, stim, arm, args))
                results.setdefault(f"{model}|{arm}", {})[stim["id"]] = cell
                print(f"[{model}][{arm}][{stim['id']}] {_cell_label(arm, cell)}  "
                      f"{json.dumps({k: v for k, v in cell.items() if k != 'titles'})[:400]}")
                sys.stdout.flush()

    print("\n=== summary: registry population per model x mandate ===")
    header = ["model|arm"] + [s["id"][:26] for s in STIMULI]
    widths = [max(len(header[0]), 24)] + [max(len(h), 14) for h in header[1:]]
    print(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    for key, row in results.items():
        arm = key.split("|", 1)[1]
        cells = [_cell_label(arm, row.get(s["id"], {"error": "missing"})) for s in STIMULI]
        print(" | ".join(v.ljust(w) for v, w in zip([key] + cells, widths)))

    print("\n=== NEGATIVE CONTROL (052 breadth fan-out): must be 'none' ===")
    control_id = STIMULI[-1]["id"]
    false_positives = 0
    for key, row in results.items():
        arm = key.split("|", 1)[1]
        if arm == "tag":
            continue
        cell = row.get(control_id, {})
        label = _cell_label(arm, cell)
        if label not in ("none", "ERR"):
            false_positives += 1
            print(f"  FALSE POSITIVE  {key}: {label} -> {cell.get('inferred')}")
        else:
            print(f"  ok              {key}: {label}")
    print(f"  false positives: {false_positives}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
