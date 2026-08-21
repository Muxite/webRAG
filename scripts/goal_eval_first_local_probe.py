"""Live ($0, local-ollama) A/B of ``merge_goal_evaluation_first_enabled`` (reason-before-verdict
ordering for merge's ``goal_achieved``).

promptbench v2 (2026-08-19, §3/§9) found the SHIPPED boolean-first ordering degenerate: 5/5 models
answered ACHIEVED on 100% of a balanced isolated set, so no paired contrast was even computable.
What that study could NOT answer is whether reason-first DISCRIMINATES better end to end, or merely
relocates the rubber stamp to a different constant. This probe answers exactly that, by feeding the
real ``MergeLeafAction.execute`` matched TRUE/FALSE merge scenarios and scoring the PAIR.

Stimulus is real captured merge state (``agent/idea_test_results/``), never synthetic. Two matched
pairs, each pair sharing one task's goal text and differing only in evidence completeness:

  122T  task 122 / dag_v2_playbook_race_20260821 / llama-3.2-3b, merge e183df66 -- the 4-child
        node from the compaction diagnosis. All four telescopes present AND resolved, FAST's
        keystone aperture in evidence. Ground truth ACHIEVED.
  122F  same run / gpt-4.1-nano, merge 34812b4e -- 2 children (Arecibo visit + a Serper-403-shaped
        search). Names appear in snippets, but no keystone and only 2 of 4 candidates checked.
        Ground truth NOT achieved. Deliberately the HARD false case: superficially on topic.
  140T  task 140 / csapi_g_nano_baseline_rep1, merge 98e77e89 -- search + visit + extract on the
        correctly disambiguated Mount Adams (New Hampshire) page; 5,793 ft in evidence.
        Ground truth ACHIEVED.
  140F  task 140 / csapi_g_flash_good_adaptive_rep1, merge 7b2fcb2b -- an on-topic-looking search
        that returned a hiking blog and a visit that yielded no page content; the elevation is
        absent. Ground truth NOT achieved.

Each case is run under both arms (flag OFF = shipped, flag ON = reason-first), on each model, for N
replicates. ``goal_achieved`` is recorded TWICE: ``raw`` as the model emitted it, and ``effective``
after ``MergeLeafAction``'s ``missing_requirements`` consistency guard (edc3f328) downgrades a
self-contradictory ``true`` -- the guard applies to both arms, so it is scored as shipped behavior
while remaining separable.

The wire is the ENGINE's own path: a real ``ConnectorLLM`` with ``LLM_PROVIDER=ollama`` +
``LLM_NUM_CTX``, i.e. ``OllamaNativeBackend`` (3bf7f604), so ``num_ctx`` is actually honored rather
than silently ignored by the OpenAI shim. Only the model endpoint is local; prompt assembly,
payload building, parsing and the guard are the shipped code.

Not a pytest test; run manually:

    PYTHONPATH=.:services:agent python3 scripts/goal_eval_first_local_probe.py \
        [--models llama3.2:3b,qwen2.5:7b,qwen2.5:14b] [--replicates 4] [--num-ctx 32768] \
        [--host http://localhost:11435] [--out scripts/_probe_out/goal_eval_first.json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "services"))
sys.path.insert(0, str(REPO / "agent"))

RESULTS = REPO / "agent" / "idea_test_results"

# (file, merge-node prefix, ground-truth goal_achieved, evidence tokens a genuinely engaged
# ``goal_evaluation`` would be expected to cite).
CASES: Dict[str, Tuple[str, str, bool, str]] = {
    "122T": (
        "dag_v2_playbook_race_20260821_122_meta-llama-llama-3.2-3b-instruct_graph_cfgcb42ed79_r1.json",
        "e183df66", True, r"arecibo|ratan|green\s*bank|\bfast\b|five[-\s]hundred|300\s*m",
    ),
    "122F": (
        "dag_v2_playbook_race_20260821_122_openai-gpt-4.1-nano_graph_cfg370b23e7_r1.json",
        "34812b4e", False, r"arecibo|ratan|green\s*bank|\bfast\b|five[-\s]hundred|300\s*m",
    ),
    "140T": (
        "csapi_g_nano_baseline_rep1_140_openai-gpt-4.1-nano_graph_r1.json",
        "98e77e89", True, r"5,?793|1,?766|new\s+hampshire|presidential",
    ),
    "140F": (
        "csapi_g_flash_good_adaptive_rep1_140_google-gemini-2.5-flash-lite_graph_r1.json",
        "7b2fcb2b", False, r"5,?793|1,?766|new\s+hampshire|presidential",
    ),
}


def _configure_env(host: str, model: str, num_ctx: int, timeout: float) -> None:
    """Point the real connector stack at the local ollama server, and only there."""
    if not re.match(r"^https?://(localhost|127\.0\.0\.1)(:\d+)?/?$", host.rstrip("/") + "/"):
        raise SystemExit(f"refusing non-local endpoint {host!r}: this probe is $0/local only")
    os.environ["LLM_PROVIDER"] = "ollama"
    os.environ["MODEL_API_URL"] = host.rstrip("/") + "/v1"
    os.environ["LLM_API_KEY"] = "ollama"
    os.environ["MODEL_NAME"] = model
    os.environ["LLM_NUM_CTX"] = str(num_ctx)
    os.environ["LLM_READ_TIMEOUT"] = str(int(timeout))
    os.environ["LLM_CONNECT_TIMEOUT"] = "10"


def _load_merge_node(filename: str, node_prefix: str) -> Dict[str, Any]:
    nodes = json.loads((RESULTS / filename).read_text())["execution"]["graph"]["nodes"]
    for nid, node in nodes.items():
        if nid.startswith(node_prefix) and (node.get("details") or {}).get("action") == "merge":
            return node
    raise SystemExit(f"merge node {node_prefix} not found in {filename}")


class _RecordingIO:
    """The real ``AgentIO`` over a real ``ConnectorLLM``, with the two calls merge makes taped."""

    def __init__(self, model: str):
        from agent.app.agent_io import AgentIO
        from agent.app.connector_llm import ConnectorLLM
        from shared.connector_config import ConnectorConfig

        self.model = model
        self._io = AgentIO(
            connector_llm=ConnectorLLM(ConnectorConfig()),
            connector_search=None, connector_http=None, connector_chroma=None,
        )
        self.connector_llm = self._io.connector_llm
        self.messages: List[Dict[str, str]] = []
        self.response = ""
        self.calls = 0

    def build_llm_payload(self, **kwargs) -> Dict[str, Any]:
        self.messages = kwargs.get("messages") or []
        return self._io.build_llm_payload(**kwargs)

    async def query_llm_with_fallback(self, payload, **kwargs) -> str:
        self.calls += 1
        self.response = await self._io.query_llm_with_fallback(payload, **kwargs) or ""
        return self.response


def _merge_graph(merge_node: Dict[str, Any]):
    """A one-merge-node graph carrying the captured node's real merged_results and goal text."""
    from agent.app.idea_dag import IdeaDag
    from agent.app.idea_policies.base import DetailKey, IdeaNodeStatus

    graph = IdeaDag(root_title="root")
    node = graph.add_child(graph.root_id(), merge_node.get("title") or "merge",
                           status=IdeaNodeStatus.PENDING)
    details = merge_node["details"]
    node.details[DetailKey.MERGED_RESULTS.value] = details["merged_results"]
    for key in ("goal", "original_goal"):
        if details.get(key):
            node.details[key] = details[key]
    return graph, node


async def _run_cell(case: str, arm_on: bool, model: str, args, io: _RecordingIO) -> Dict[str, Any]:
    from agent.app.idea_dag_settings import load_idea_dag_settings
    from agent.app.idea_policies.actions import MergeLeafAction

    filename, prefix, expected, evidence_rx = CASES[case]
    graph, node = _merge_graph(_load_merge_node(filename, prefix))
    settings = load_idea_dag_settings()
    settings["merge_goal_evaluation_first_enabled"] = arm_on
    settings["merge_model"] = model
    settings["llm_timeout_seconds"] = args.timeout

    started = time.perf_counter()
    error = ""
    try:
        result = await MergeLeafAction(settings=settings).execute(graph, node.node_id, io)
    except Exception as exc:  # noqa: BLE001
        result, error = {}, f"{type(exc).__name__}: {exc}"

    system_msg = "".join(m.get("content", "") for m in io.messages if m.get("role") == "system")
    prompt_chars = sum(len(m.get("content", "")) for m in io.messages)
    synthesized = (result or {}).get("synthesized") or {}
    raw = synthesized.get("goal_achieved") if isinstance(synthesized, dict) else None
    evaluation = (result or {}).get("goal_evaluation") or ""
    missing = (result or {}).get("missing_requirements") or []
    effective = (result or {}).get("goal_achieved")
    return {
        "case": case, "expected": expected, "arm": "on" if arm_on else "off", "model": model,
        "error": error, "wall_s": round(time.perf_counter() - started, 1),
        "prompt_chars": prompt_chars,
        # The flag's two ordering sources both live in the system message; assert it actually moved.
        "reason_first_in_prompt": system_msg.find("goal_evaluation") < system_msg.find("goal_achieved"),
        "raw_goal_achieved": raw,
        "effective_goal_achieved": effective,
        "raw_correct": (bool(raw) == expected) if raw is not None else None,
        "effective_correct": (bool(effective) == expected) if effective is not None else None,
        "downgraded_by_guard": bool(raw) and not bool(effective),
        "missing_requirements": missing,
        "goal_evaluation": evaluation,
        "eval_chars": len(evaluation),
        # Cheap proxy for "engaged with THIS evidence" vs a generic justification.
        "eval_cites_evidence": bool(re.search(evidence_rx, evaluation, re.IGNORECASE)),
        "summary": (synthesized.get("summary") if isinstance(synthesized, dict) else "") or "",
        "parse_failed": evaluation == "Failed to parse LLM response",
        "action_error": (result or {}).get("error") or "",
        "raw_response": io.response,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="llama3.2:3b,qwen2.5:7b,qwen2.5:14b")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--arms", default="off,on")
    parser.add_argument("--host", default="http://localhost:11435")
    parser.add_argument("--replicates", type=int, default=4)
    # Replicate index to start at, so a top-up run can be merged with an earlier file
    # without colliding on ``rep`` (the analyzer pairs cases by it).
    parser.add_argument("--rep-offset", type=int, default=0)
    parser.add_argument("--num-ctx", type=int, default=32768)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", default="scripts/_probe_out/goal_eval_first.json")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    arms = [a.strip() == "on" for a in args.arms.split(",") if a.strip()]
    rows: List[Dict[str, Any]] = []
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for model in models:
        _configure_env(args.host, model, args.num_ctx, args.timeout)
        io = _RecordingIO(model)
        for rep in range(args.replicates):
            for case in cases:
                for arm_on in arms:
                    row = await _run_cell(case, arm_on, model, args, io)
                    row["rep"] = rep + args.rep_offset
                    rows.append(row)
                    print(f"[{model} rep{rep} {case} arm={row['arm']}] "
                          f"expect={row['expected']} raw={row['raw_goal_achieved']} "
                          f"eff={row['effective_goal_achieved']} "
                          f"missing={len(row['missing_requirements'])} "
                          f"cites={row['eval_cites_evidence']} "
                          f"prompt={row['prompt_chars']} {row['wall_s']}s {row['error']}",
                          flush=True)
                    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
