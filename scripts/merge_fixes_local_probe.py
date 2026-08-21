"""Live ($0, local-ollama) probe of the two merge fixes: 5601309e (compaction reaches the nested
``result`` payload) and 59e29494 (``should_create_merge_node`` requires >=2 substantive children).

The offline tests and the real-data replays already prove the prompt SHRINKS and the gate FIRES.
This probe answers the remaining question: driven by a REAL model, does either fix change what the
merge synthesis actually SAYS?

Stimulus is not synthetic. Both cases are lifted verbatim out of the diagnosed run
``dag_v2_playbook_race_20260821`` (``agent/idea_test_results/``):

  A (compaction)  task 122 / llama-3.2-3b / graph / cfgcb42ed79, merge node e183df66 -- the
                  "celebrated" 0.80 cell. 4 DONE children: Arecibo 58,647 chars, RATAN-600 6,271,
                  Green Bank 16,072, FAST 26,338 (the correct answer). 665,962 chars serialized.
                  Two arms, differing ONLY in the compaction function:
                    pre   the exact pre-5601309e loop (copied from the parent commit), which
                          matched nothing, so the blob hit ``merge_max_json_chars`` and was
                          blind-truncated mid-JSON with only Arecibo visible.
                    post  the shipped recursive ``_compact_merged_results``.
                  Everything else -- prompts, settings, goal text, model, decoding params -- is
                  byte-identical between arms.

  B (gate)        task 122 / gpt-4.1-nano / graph / cfg370b23e7, merge node 34812b4e -- 2 formal
                  children, one real Arecibo visit (20,084 chars) and one Serper-403-shaped search
                  recorded ``success: true, content: "", results: []``. Runs
                  ``should_create_merge_node`` with the kill switch ON and OFF, and when the gate
                  lets it through, makes the real synthesis call so the wasted completion can be
                  read rather than assumed.

The wire is ollama's NATIVE ``/api/chat`` rather than its OpenAI-compatible shim, because the shim
ignores ``num_ctx`` and ollama's 4096-token default would silently truncate the ``pre`` arm a
SECOND time inside the server -- which would hide the very effect under test. Both arms get the
same generous ``num_ctx``. Message content is exactly what ``MergeLeafAction`` builds.

Not a pytest test; run manually:

    PYTHONPATH=.:services:agent python3 scripts/merge_fixes_local_probe.py \
        [--models llama3.2:3b] [--replicates 5] [--num-ctx 32768] \
        [--host http://localhost:11435] [--out scripts/_probe_out/merge_fixes.json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "services"))
sys.path.insert(0, str(REPO / "agent"))

import httpx  # noqa: E402

from agent.app.idea_dag import IdeaDag  # noqa: E402
from agent.app.idea_dag_settings import load_idea_dag_settings  # noqa: E402
from agent.app.idea_policies.actions import MergeLeafAction  # noqa: E402
from agent.app.idea_policies.base import DetailKey, IdeaNodeStatus  # noqa: E402
from agent.app.idea_policies.merge import SimpleMergePolicy  # noqa: E402
from agent.app.idea_tests import test_122_tier5_filled_aperture_telescope_survivor as T122  # noqa: E402

RESULTS = REPO / "agent" / "idea_test_results"
CASE_A = (RESULTS / "dag_v2_playbook_race_20260821_122_meta-llama-llama-3.2-3b-instruct"
                    "_graph_cfgcb42ed79_r1.json", "e183df66-863e-4d9d-b2f5-34f089665246")
CASE_B = (RESULTS / "dag_v2_playbook_race_20260821_122_openai-gpt-4.1-nano"
                    "_graph_cfg370b23e7_r1.json", "34812b4e")


def _load_merge_node(path: Path, node_prefix: str) -> Dict[str, Any]:
    nodes = json.loads(path.read_text())["execution"]["graph"]["nodes"]
    for nid, node in nodes.items():
        if nid.startswith(node_prefix) and (node.get("details") or {}).get("action") == "merge":
            return node
    raise SystemExit(f"merge node {node_prefix} not found in {path.name}")


# ---------------------------------------------------------------------------------------------
# The pre-5601309e compaction, copied verbatim from ``git show 5601309e^`` so the control arm
# cannot drift from what actually shipped in the diagnosed run.
# ---------------------------------------------------------------------------------------------
def _compact_pre_fix(merged_results: Any) -> Any:
    compacted = []
    for mr in merged_results:
        if isinstance(mr, dict):
            c: Dict[str, Any] = {}
            for k, v in mr.items():
                if k in ("content", "content_full", "content_with_links", "links_full",
                         "link_contexts", "_links_inline"):
                    if k == "content" and isinstance(v, str):
                        c[k] = v[:2000] + "..." if len(v) > 2000 else v
                    continue
                elif isinstance(v, str) and len(v) > 5000:
                    c[k] = v[:5000] + "..."
                else:
                    c[k] = v
            compacted.append(c)
        else:
            compacted.append(mr)
    return compacted


class _OllamaIO:
    """The two ``AgentIO`` methods ``MergeLeafAction.execute`` calls, backed by local ollama.

    Records every payload so prompt size / call count are measured, not inferred.
    """

    def __init__(self, host: str, model: str, num_ctx: int, num_predict: int, timeout: float):
        self.host, self.model = host.rstrip("/"), model
        self.num_ctx, self.num_predict, self.timeout = num_ctx, num_predict, timeout
        self.payloads: List[Dict[str, Any]] = []
        self.calls = 0
        self.usage: List[Dict[str, Any]] = []

    def build_llm_payload(self, messages=None, **kwargs) -> Dict[str, Any]:
        payload = {"messages": messages, **kwargs}
        self.payloads.append(payload)
        return payload

    async def query_llm_with_fallback(self, payload, model_name=None, fallback_model=None,
                                      timeout_seconds=None) -> str:
        self.calls += 1
        body = {
            "model": self.model,
            "messages": payload["messages"],
            "stream": False,
            "format": "json" if payload.get("json_mode") else None,
            "options": {
                "temperature": payload.get("temperature", 0.3),
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }
        body = {k: v for k, v in body.items() if v is not None}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.host}/api/chat", json=body)
            resp.raise_for_status()
            data = resp.json()
        self.usage.append({k: data.get(k) for k in
                           ("prompt_eval_count", "eval_count", "total_duration")})
        return (data.get("message") or {}).get("content", "")


_ABSTAIN_RE = re.compile(r"not (?:available|provided|found|specified|given|stated)"
                         r"|cannot be determined|unable to", re.IGNORECASE)


def _score_synthesis(text: str, prompt: str) -> Dict[str, Any]:
    """Which of the four telescopes the synthesis actually discusses, and whether it lands the
    keystone. ``prompt`` is scored the same way so "the model never saw it" and "the model saw it
    and dropped it" stay distinguishable."""
    def named(hay: str) -> List[str]:
        return [c["key"] for c in T122.CANDIDATES if re.search(c["name_rx"], hay, re.IGNORECASE)]

    def resolved(hay: str) -> List[str]:
        return [c["key"] for c in T122.CANDIDATES
                if re.search(c["name_rx"], hay, re.IGNORECASE)
                and re.search(c["prop_rx"], hay, re.IGNORECASE)]

    out_named = named(text)
    return {
        "prompt_telescopes": named(prompt),
        "telescopes_named": out_named,
        "n_telescopes_named": len(out_named),
        "telescopes_resolved": resolved(text),
        "keystone_300m": bool(T122.KEYSTONE_RX.search(text or "")),
        "elects_fast": bool(re.search(r"\bfast\b|five[-\s]hundred|aperture\s+spherical",
                                      text or "", re.IGNORECASE)),
        "abstained": bool(_ABSTAIN_RE.search(text or "")),
    }


def _merge_graph(merge_node: Dict[str, Any]) -> tuple:
    """A one-merge-node graph carrying the captured node's real merged_results and goal text."""
    graph = IdeaDag(root_title="root")
    node = graph.add_child(graph.root_id(), merge_node.get("title") or "merge",
                           status=IdeaNodeStatus.PENDING)
    details = merge_node["details"]
    node.details[DetailKey.MERGED_RESULTS.value] = details["merged_results"]
    for key in ("goal", "original_goal"):
        if details.get(key):
            node.details[key] = details[key]
    return graph, node


async def _run_case_a(arm: str, model: str, args, merge_node: Dict[str, Any]) -> Dict[str, Any]:
    graph, node = _merge_graph(merge_node)
    io = _OllamaIO(args.host, model, args.num_ctx, args.num_predict, args.timeout)
    action = MergeLeafAction(settings=load_idea_dag_settings())
    original = MergeLeafAction._compact_merged_results
    if arm == "pre":
        MergeLeafAction._compact_merged_results = staticmethod(_compact_pre_fix)
    started = time.perf_counter()
    error = ""
    try:
        result = await action.execute(graph, node.node_id, io)
    except Exception as exc:  # noqa: BLE001
        result, error = {}, f"{type(exc).__name__}: {exc}"
    finally:
        MergeLeafAction._compact_merged_results = original

    prompt = ""
    if io.payloads:
        prompt = "".join(m.get("content", "") for m in io.payloads[-1]["messages"])
    synthesized = (result or {}).get("synthesized") or {}
    text = json.dumps(synthesized, ensure_ascii=False) if isinstance(synthesized, dict) else str(synthesized)
    return {
        "case": "A_compaction", "arm": arm, "model": model, "error": error,
        "wall_s": round(time.perf_counter() - started, 1),
        "llm_calls": io.calls,
        "prompt_chars": len(prompt),
        "prompt_truncated": "...[truncated]" in prompt,
        "usage": io.usage,
        "goal_achieved": (result or {}).get("goal_achieved"),
        "parse_failed": (result or {}).get("goal_evaluation") == "Failed to parse LLM response",
        "summary": (synthesized.get("summary") if isinstance(synthesized, dict) else "") or "",
        "key_findings": (synthesized.get("key_findings") if isinstance(synthesized, dict) else []) or [],
        **_score_synthesis(text, prompt),
    }


async def _run_case_b(gate_on: bool, model: str, args, merge_node: Dict[str, Any]) -> Dict[str, Any]:
    """The gate decides on the PARENT's children, so rebuild the parent shape: each captured
    merged child becomes a real node carrying its own ``action_result``."""
    settings = load_idea_dag_settings()
    settings["merge_require_substantive_children_enabled"] = gate_on
    graph = IdeaDag(root_title="root")
    parent = graph.add_child(graph.root_id(), "parent", status=IdeaNodeStatus.ACTIVE)
    children = merge_node["details"]["merged_results"]
    for entry in children:
        child = graph.add_child(parent.node_id, entry.get("title") or "child",
                                status=IdeaNodeStatus.DONE)
        child.details[DetailKey.ACTION_RESULT.value] = entry.get("result")
    policy = SimpleMergePolicy(settings=settings)
    substantive = policy._substantive_child_count(graph, parent)
    would_merge = policy.should_create_merge_node(graph, parent.node_id)

    row: Dict[str, Any] = {
        "case": "B_gate", "arm": f"gate_{'on' if gate_on else 'off'}", "model": model,
        "formal_children": len(parent.children), "substantive_children": substantive,
        "would_merge": would_merge, "llm_calls": 0, "error": "",
    }
    if not would_merge:
        row["summary"] = ""
        return row

    graph2, node = _merge_graph(merge_node)
    io = _OllamaIO(args.host, model, args.num_ctx, args.num_predict, args.timeout)
    started = time.perf_counter()
    try:
        result = await MergeLeafAction(settings=settings).execute(graph2, node.node_id, io)
    except Exception as exc:  # noqa: BLE001
        result, row["error"] = {}, f"{type(exc).__name__}: {exc}"
    prompt = "".join(m.get("content", "") for m in io.payloads[-1]["messages"]) if io.payloads else ""
    synthesized = (result or {}).get("synthesized") or {}
    text = json.dumps(synthesized, ensure_ascii=False) if isinstance(synthesized, dict) else str(synthesized)
    row.update({
        "llm_calls": io.calls, "wall_s": round(time.perf_counter() - started, 1),
        "prompt_chars": len(prompt), "usage": io.usage,
        "goal_achieved": (result or {}).get("goal_achieved"),
        "summary": (synthesized.get("summary") if isinstance(synthesized, dict) else "") or "",
        **_score_synthesis(text, prompt),
    })
    return row


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="llama3.2:3b")
    parser.add_argument("--cases", default="A,B")
    parser.add_argument("--arms", default="pre,post")
    parser.add_argument("--host", default="http://localhost:11435")
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--num-ctx", type=int, default=32768)
    parser.add_argument("--num-predict", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--out", default="scripts/_probe_out/merge_fixes.json")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cases = [c.strip().upper() for c in args.cases.split(",") if c.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    node_a = _load_merge_node(*CASE_A)
    node_b = _load_merge_node(*CASE_B)
    rows: List[Dict[str, Any]] = []
    for model in models:
        for rep in range(args.replicates):
            if "A" in cases:
                for arm in arms:
                    row = await _run_case_a(arm, model, args, node_a)
                    row["rep"] = rep
                    rows.append(row)
                    print(f"[A {model} rep{rep} {arm}] prompt={row['prompt_chars']} "
                          f"trunc={row['prompt_truncated']} "
                          f"prompt_scopes={row['prompt_telescopes']} "
                          f"out_scopes={row['telescopes_named']} "
                          f"keystone={row['keystone_300m']} goal={row['goal_achieved']} "
                          f"{row['wall_s']}s {row['error']}", flush=True)
            if "B" in cases:
                for gate_on in (True, False):
                    row = await _run_case_b(gate_on, model, args, node_b)
                    row["rep"] = rep
                    rows.append(row)
                    print(f"[B {model} rep{rep} gate={'on' if gate_on else 'off'}] "
                          f"substantive={row['substantive_children']}/{row['formal_children']} "
                          f"would_merge={row['would_merge']} calls={row['llm_calls']} "
                          f"{row.get('error','')}", flush=True)
            out_path = REPO / args.out
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
