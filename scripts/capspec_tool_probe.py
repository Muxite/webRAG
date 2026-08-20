#!/usr/bin/env python3
"""$0 capability probe: which local models can do (a) OpenAI tool-calling, (b) raw JSON emission?

Arm A (langgraph_react) needs (a) — create_react_agent binds tools via the function-calling API.
Arm B (graph / DAG v2) needs only (b) — the engine parses JSON out of plain text itself.

Runs against badmodel-ollama on :11435. Smallest models first so GPU swaps stay cheap.
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error

BASE = "http://localhost:11435/v1"
MODELS = [
    "qwen2.5:0.5b",
    "tinyllama:latest",
    "llama3.2:1b",
    "qwen2.5:1.5b",
    "gemma2:2b",
    "phi3:mini",
    "llama3.2:3b",
    "qwen2.5:7b",
    "llama3.1:8b",
    "qwen2.5:14b",
]

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search",
        "description": "Web search. Returns titles, URLs and snippets for the query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "search keywords"}},
            "required": ["query"],
        },
    },
}]

TOOL_PROMPT = (
    "You are a web-research agent. Use the search tool to gather evidence before answering; "
    "never answer from memory.\n\nTASK: What is the height of the Huajiang Grand Canyon Bridge?"
)

JSON_PROMPT = (
    "Respond with ONLY a JSON object, no prose, no markdown fences. "
    'Schema: {"action": "search", "query": "<short keywords>", "reason": "<one sentence>"}\n\n'
    "TASK: Find the height of the Huajiang Grand Canyon Bridge. What is your first action?"
)


def post(payload, timeout=180):
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer ollama"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def probe_tools(model):
    t0 = time.perf_counter()
    try:
        d = post({
            "model": model,
            "messages": [{"role": "user", "content": TOOL_PROMPT}],
            "tools": TOOLS,
            "temperature": 0.1,
            "max_tokens": 512,
        })
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300]
        return {"ok": False, "why": f"HTTP {e.code}: {body}", "secs": round(time.perf_counter() - t0, 1)}
    except Exception as e:
        return {"ok": False, "why": f"{type(e).__name__}: {e}", "secs": round(time.perf_counter() - t0, 1)}
    msg = (d.get("choices") or [{}])[0].get("message", {}) or {}
    calls = msg.get("tool_calls") or []
    secs = round(time.perf_counter() - t0, 1)
    if not calls:
        return {"ok": False, "why": "no tool_calls emitted",
                "content": (msg.get("content") or "")[:160], "secs": secs}
    fn = (calls[0].get("function") or {})
    args_raw = fn.get("arguments")
    try:
        args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        args_ok = isinstance(args, dict) and bool(str(args.get("query", "")).strip())
    except Exception:
        args_ok = False
    return {"ok": bool(fn.get("name") == "search" and args_ok), "why": "",
            "name": fn.get("name"), "args": str(args_raw)[:120], "n_calls": len(calls), "secs": secs}


def probe_json(model):
    t0 = time.perf_counter()
    try:
        d = post({
            "model": model,
            "messages": [{"role": "user", "content": JSON_PROMPT}],
            "temperature": 0.1,
            "max_tokens": 512,
        })
    except Exception as e:
        return {"ok": False, "why": f"{type(e).__name__}: {e}", "secs": round(time.perf_counter() - t0, 1)}
    text = ((d.get("choices") or [{}])[0].get("message", {}) or {}).get("content") or ""
    secs = round(time.perf_counter() - t0, 1)
    # Same salvage the engine does: strip fences, take the outermost {...}
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```")[1] if len(s.split("```")) > 1 else s
        s = s[4:] if s.lower().startswith("json") else s
    i, j = s.find("{"), s.rfind("}")
    ok, why = False, ""
    if i >= 0 and j > i:
        try:
            obj = json.loads(s[i:j + 1])
            ok = isinstance(obj, dict) and bool(str(obj.get("query", "")).strip())
            why = "" if ok else f"parsed but wrong keys: {list(obj)[:5]}"
        except Exception as e:
            why = f"unparseable: {e}"
    else:
        why = "no JSON object in output"
    return {"ok": ok, "why": why, "raw": text[:160].replace("\n", " "), "secs": secs}


def main():
    rows = []
    for m in MODELS:
        print(f"\n=== {m} ===", flush=True)
        t = probe_tools(m)
        print(f"  tool_calling : {'PASS' if t['ok'] else 'FAIL'}  ({t['secs']}s)  {t.get('why','')}"
              f"{'  args=' + t.get('args', '') if t['ok'] else ''}", flush=True)
        if not t["ok"] and t.get("content"):
            print(f"                 content-instead: {t['content']!r}", flush=True)
        j = probe_json(m)
        print(f"  json_emission: {'PASS' if j['ok'] else 'FAIL'}  ({j['secs']}s)  {j.get('why','')}", flush=True)
        if not j["ok"]:
            print(f"                 raw: {j.get('raw','')!r}", flush=True)
        rows.append({"model": m, "tools": t, "json": j})
    print("\n\n=== SUMMARY ===")
    print(f"{'model':20s} {'langgraph-able':16s} {'dagv2-able':12s} {'t_tool':>8s} {'t_json':>8s}")
    for r in rows:
        print(f"{r['model']:20s} {('YES' if r['tools']['ok'] else 'NO'):16s} "
              f"{('YES' if r['json']['ok'] else 'NO'):12s} "
              f"{r['tools']['secs']:>8} {r['json']['secs']:>8}")
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "agent", "idea_test_results", "capspec_tool_probe.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
