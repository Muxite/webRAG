"""Live ($0, local-ollama) probe of the ``num_ctx`` truncation hypothesis
(``agent/app/ENGINE_DESIGN_REVIEW.md`` PART 1).

The claim under test: ``ConnectorLLM`` never sets ``num_ctx``, so ollama serves the graph
engine's long prompts at whatever context the server defaults to, and anything past that
ceiling is silently dropped server-side -- which would depress ``graph`` (long prompts)
without touching ``seq_react``/``langgraph`` (short prompts) on the same model.

Stimulus is real, not synthetic: the merge prompt ``MergeLeafAction`` actually builds from the
captured merge node of the exact cell the review cites --
``csloc_g_q7_baseline_rep1_122_qwen2.5:7b_graph_r1.json``, score 0.15 -- padded, when a bigger
size is requested, with more of that same run's real visit text.

A tail probe rides on the real prompt: a marker near the HEAD and one near the TAIL, plus the
answer instruction. Both markers are needed for a correct answer, so "the model saw the whole
prompt" is measured rather than assumed, and ollama's own ``prompt_eval_count`` is compared to
the true token count from ``/api/tokenize`` so truncation is visible on the wire too.

Three arms per size:

  shim            POST /v1/chat/completions -- byte-identical in shape to what
                  ``OpenAICompatibleBackend`` sends today (no options at all).
  native_default  POST /api/chat with no ``num_ctx`` -- isolates the server default.
  native_32k      POST /api/chat with ``num_ctx`` pinned -- the proposed fix.

Not a pytest test; run manually:

    PYTHONPATH=.:services:agent python3 scripts/ollama_num_ctx_probe.py \\
        [--models qwen2.5:7b] [--sizes 8000,20000,30000] [--replicates 3] \\
        [--host http://localhost:11435] [--out scripts/_probe_out/num_ctx.json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "services"))
sys.path.insert(0, str(REPO / "agent"))

import httpx  # noqa: E402

from agent.app.idea_dag import IdeaDag  # noqa: E402
from agent.app.idea_dag_settings import load_idea_dag_settings  # noqa: E402
from agent.app.idea_policies.actions import MergeLeafAction  # noqa: E402
from agent.app.idea_policies.base import DetailKey, IdeaNodeStatus  # noqa: E402

RESULTS = REPO / "agent" / "idea_test_results"
CELL = RESULTS / "csloc_g_q7_baseline_rep1_122_qwen2.5:7b_graph_r1.json"

HEAD_MARKER = "ARCHIVE-NOTE-HEAD: the archivist's codeword is FALCON-7731."
TAIL_MARKER = "ARCHIVE-NOTE-TAIL: the librarian's codeword is BADGER-9042."
ASK = (
    "\n\nBefore anything else: two ARCHIVE-NOTE lines appear in the material above, one near "
    "the beginning and one near the end. Reply with JSON only, no commentary, exactly: "
    '{"head_codeword": "...", "tail_codeword": "..."} using the codeword from each note. '
    'If a note is not present in what you were given, use "MISSING" for it.'
)


def _cell_nodes() -> Dict[str, Any]:
    return json.loads(CELL.read_text())["execution"]["graph"]["nodes"]


def _merge_node(nodes: Dict[str, Any]) -> Dict[str, Any]:
    for node in nodes.values():
        if (node.get("details") or {}).get("action") == "merge":
            return node
    raise SystemExit("no merge node in the captured cell")


def _visit_texts(nodes: Dict[str, Any]) -> List[str]:
    out = []
    for node in nodes.values():
        ar = (node.get("details") or {}).get("action_result") or {}
        text = ar.get("content_full") or ar.get("content") or ""
        if text:
            out.append(text)
    return out


class _Capture:
    """Stands in for ``AgentIO`` so ``MergeLeafAction`` builds its real prompt without a wire."""

    def __init__(self) -> None:
        self.payloads: List[Dict[str, Any]] = []

    def build_llm_payload(self, messages=None, **kwargs) -> Dict[str, Any]:
        payload = {"messages": messages, **kwargs}
        self.payloads.append(payload)
        return payload

    async def query_llm_with_fallback(self, payload, **kwargs) -> str:
        return "{}"


async def _real_merge_prompt() -> str:
    nodes = _cell_nodes()
    merge_node = _merge_node(nodes)
    graph = IdeaDag(root_title="root")
    node = graph.add_child(graph.root_id(), merge_node.get("title") or "merge",
                           status=IdeaNodeStatus.PENDING)
    details = merge_node["details"]
    node.details[DetailKey.MERGED_RESULTS.value] = details.get("merged_results") or []
    for key in ("goal", "original_goal"):
        if details.get(key):
            node.details[key] = details[key]
    io = _Capture()
    await MergeLeafAction(settings=load_idea_dag_settings()).execute(graph, node.node_id, io)
    if not io.payloads:
        raise SystemExit("merge action built no payload")
    return "\n\n".join(m.get("content", "") for m in io.payloads[-1]["messages"])


async def _tokenize(client: httpx.AsyncClient, host: str, model: str, text: str) -> int:
    resp = await client.post(f"{host}/api/tokenize", json={"model": model, "text": text})
    if resp.status_code == 404:  # older ollama: no tokenize endpoint
        return -1
    resp.raise_for_status()
    return len(resp.json().get("tokens") or [])


def _build_prompt(base: str, filler: List[str], target_chars: int) -> str:
    body = base
    idx = 0
    while len(body) < target_chars and filler:
        body += "\n\n[additional captured page text]\n" + filler[idx % len(filler)]
        idx += 1
    body = body[:target_chars]
    return f"{HEAD_MARKER}\n\n{body}\n\n{TAIL_MARKER}{ASK}"


async def _call(client: httpx.AsyncClient, host: str, model: str, arm: str,
                prompt: str, num_ctx: int, nonce: str = "") -> Dict[str, Any]:
    # A per-call nonce at the very front busts ollama's prefix cache, so ``prompt_eval_count``
    # keeps reporting the tokens actually evaluated instead of collapsing to the cache miss.
    prompt = f"[request {nonce}]\n{prompt}" if nonce else prompt
    started = time.perf_counter()
    if arm == "shim":
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0, "response_format": {"type": "json_object"}}
        resp = await client.post(f"{host}/v1/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        prompt_eval = (data.get("usage") or {}).get("prompt_tokens")
    else:
        options: Dict[str, Any] = {"temperature": 0.0, "num_predict": 128}
        if arm != "native_default":
            options["num_ctx"] = num_ctx
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "stream": False, "format": "json", "options": options}
        resp = await client.post(f"{host}/api/chat", json=body)
        resp.raise_for_status()
        data = resp.json()
        content = (data.get("message") or {}).get("content", "")
        prompt_eval = data.get("prompt_eval_count")
    try:
        parsed = json.loads(content or "{}")
    except (TypeError, ValueError):
        parsed = {}
    head = str(parsed.get("head_codeword") or "").upper()
    tail = str(parsed.get("tail_codeword") or "").upper()
    return {
        "arm": arm,
        "prompt_eval_count": prompt_eval,
        "head_ok": "FALCON-7731" in head,
        "tail_ok": "BADGER-9042" in tail,
        "head_raw": head[:40],
        "tail_raw": tail[:40],
        "wall_s": round(time.perf_counter() - started, 1),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="qwen2.5:7b")
    parser.add_argument("--sizes", default="8000,60000,90000",
                        help="target prompt sizes in CHARS (before markers)")
    parser.add_argument("--arms", default="shim,native_default,native_32k")
    parser.add_argument("--host", default="http://localhost:11435")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--num-ctx", type=int, default=32768)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--out", default="scripts/_probe_out/num_ctx.json")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    base = await _real_merge_prompt()
    filler = _visit_texts(_cell_nodes())
    print(f"real merge prompt: {len(base)} chars; {len(filler)} captured pages as filler",
          flush=True)

    rows: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        host = args.host.rstrip("/")
        for model in models:
            for size in sizes:
                prompt = _build_prompt(base, filler, size)
                true_tokens = await _tokenize(client, host, model, prompt)
                for rep in range(args.replicates):
                    for arm in arms:
                        row = await _call(client, host, model, arm, prompt, args.num_ctx,
                                          nonce=f"{size}-{rep}-{arm}-{time.time():.3f}")
                        row.update({"model": model, "target_chars": size,
                                    "prompt_chars": len(prompt), "true_tokens": true_tokens,
                                    "rep": rep})
                        rows.append(row)
                        print(f"[{model} {size}c/{true_tokens}t rep{rep} {arm}] "
                              f"eval={row['prompt_eval_count']} head={row['head_ok']} "
                              f"tail={row['tail_ok']} {row['wall_s']}s", flush=True)
                out_path = REPO / args.out
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
