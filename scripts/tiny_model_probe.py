#!/usr/bin/env python3
"""Diagnose why the smallest local models read ZERO pages, without running a benchmark cell.

Three sub-problem harnesses, all offline-ish ($0, local Ollama only), each isolating one link in
the chain between "the agent sends a prompt" and "the model calls a tool":

``ceiling``
    The context window each model ACTUALLY gets, measured by sending a prompt far larger than the
    window and reading back ``usage.prompt_tokens``. This is the number that matters, and it is
    NOT the ``num_ctx`` the run requests: ``langgraph_solver._build_llm`` builds a plain
    ``ChatOpenAI`` against ollama's ``/v1`` shim and never sends ``options.num_ctx``, so the
    served window is ``min(model's trained context, OLLAMA_CONTEXT_LENGTH)``. A model whose
    ceiling is below the transcript size has its prompt truncated AT THE HEAD — dropping the
    system message that carries the whole tool protocol.

``sweep``
    For one model, the fill level at which it stops emitting a parseable action. Sends the REAL
    ``_SYSTEM`` + ``build_protocol`` system message plus a growing wall of filler "page text", and
    runs each completion through the real ``extract_decision``. Reproduces gemma2:2b's bare-fence
    failure exactly: parseable below the ceiling, a 3-character ``` at it.

``cells``
    Categorical scoreboard over stored result JSONs: per model, how many cells READ AT LEAST ONE
    PAGE and how many scored above 0.000. Written because pages-read is the signal that survives
    at this suite's n; mean score is not.

Usage::

    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/tiny_model_probe.py ceiling
    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/tiny_model_probe.py sweep --model gemma2:2b
    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/tiny_model_probe.py cells 'ladder02_*_r1.json'
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

DEFAULT_API = "http://127.0.0.1:11435/v1"
DEFAULT_MODELS = (
    "qwen2.5:7b", "llama3.2:3b", "gemma2:2b", "phi3:mini", "qwen2.5:1.5b",
    "qwen2.5:0.5b", "tinyllama",
)
#: Filler shaped like a visited page, so the sweep loads context the way a real transcript does.
_FILLER = (
    "SOURCE: https://en.wikipedia.org/wiki/Filler_Page\n"
    "The page text continues with irrelevant detail about geography and history. "
)


def _chat(api: str, model: str, messages: List[Dict[str, str]],
          max_tokens: Optional[int] = None, temperature: float = 0.1) -> Tuple[str, int]:
    """One ``/v1/chat/completions`` call. Returns ``(content, prompt_tokens)``."""
    body: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    req = urllib.request.Request(
        api.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer dummy"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"], int(data["usage"]["prompt_tokens"])


def _system_prompt() -> str:
    """The EXACT system message the emulated transport sends (``_SYSTEM`` + tool protocol)."""
    import agent.app.langgraph_solver as solver
    from agent.app.prompted_tools import ToolSpec, build_protocol

    specs = [
        ToolSpec(name="search", description="Search the web for a query.", arg_names=("query",)),
        ToolSpec(name="visit", description="Open a URL and read its text.", arg_names=("url",)),
    ]
    return f"{solver._SYSTEM}\n{build_protocol(specs)}"


def cmd_ceiling(args: argparse.Namespace) -> int:
    """Measure the context window each model is actually served through the ``/v1`` shim."""
    probe = "word " * 40000
    print(f"{'model':16}{'served_ceiling_tokens':>22}")
    for model in args.models:
        try:
            _, served = _chat(args.api, model, [{"role": "user", "content": probe}], max_tokens=1)
        except Exception as exc:  # noqa: BLE001
            print(f"{model:16}{'ERROR: ' + str(exc)[:40]:>22}")
            continue
        print(f"{model:16}{served:>22}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Find the fill level at which one model stops emitting a parseable action."""
    from agent.app.prompted_tools import extract_decision

    system = _system_prompt()
    tail = ("\nTASK: Find the height of the Inco Superstack in metres.\n\n"
            "Return the next step as JSON.")
    print(f"system+protocol = {len(system)} chars")
    print(f"{'filler_words':>13}{'prompt_tokens':>15}{'parsed':>8}  completion_head")
    for words in args.fill:
        body = (_FILLER * max(1, words // 20))[: words * 6]
        try:
            text, served = _chat(args.api, args.model,
                                 [{"role": "system", "content": system},
                                  {"role": "user", "content": body + tail}])
        except Exception as exc:  # noqa: BLE001
            print(f"{words:>13}{'ERROR':>15}  {str(exc)[:60]}")
            continue
        ok = extract_decision(text).value is not None
        print(f"{words:>13}{served:>15}{str(ok):>8}  {text[:70]!r}")
    return 0


def cmd_cells(args: argparse.Namespace) -> int:
    """Categorical scoreboard over stored result JSONs: pages read, and cells above 0.000."""
    root = Path(args.results_dir)
    agg: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for path in sorted(glob.glob(str(root / args.pattern))):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        checks = data.get("validation", {}).get("grep_validations") or []
        if not checks:
            continue
        score = sum(c.get("score", 0.0) for c in checks) / len(checks)
        visits = int(((data.get("execution", {}).get("observability", {})
                       .get("visit", {})).get("count", 0)) or 0)
        row = agg[data.get("model", "?")]
        row[0] += 1
        row[1] += 1 if visits > 0 else 0
        row[2] += 1 if score > 0 else 0
        row[3] += visits
    if not agg:
        print(f"no result files matched {args.pattern!r} under {root}")
        return 1
    print(f"{'model':16}{'cells':>7}{'read>=1 page':>14}{'score>0':>9}{'visits':>8}")
    for model, row in sorted(agg.items()):
        n = int(row[0])
        print(f"{model:16}{n:>7}{int(row[1]):>9}/{n:<4}{int(row[2]):>5}/{n:<3}{int(row[3]):>8}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default=DEFAULT_API, help="OpenAI-compatible base URL")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ceil = sub.add_parser("ceiling", help="measured context window per model")
    p_ceil.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    p_ceil.set_defaults(func=cmd_ceiling)

    p_sweep = sub.add_parser("sweep", help="fill level at which a model stops emitting an action")
    p_sweep.add_argument("--model", default="gemma2:2b")
    p_sweep.add_argument("--fill", nargs="+", type=int,
                         default=[500, 3000, 5000, 7000, 9000, 12000])
    p_sweep.set_defaults(func=cmd_sweep)

    p_cells = sub.add_parser("cells", help="categorical scoreboard over stored results")
    p_cells.add_argument("pattern", nargs="?", default="*_r1.json")
    p_cells.add_argument("--results-dir", default="agent/idea_test_results")
    p_cells.set_defaults(func=cmd_cells)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
