"""Live ($0, local-ollama) probe of the three behavioral fixes in commit 50f25a34
("fix the langgraph forced-synthesis invariant and mask search-backend outages distinctly from
empty results", ``agent/app/langgraph_solver.py``).

The unit tests in ``agent/tests/langgraph_solver_test.py`` prove each code path fires against a
stubbed graph. This probe answers the different question: under the REAL
``create_react_agent`` loop driven by a real (small, local) model, do the induced conditions
actually arise, and does the fix change the outcome?

  A. Step exhaustion (fix 1). Under-budget a genuinely 3-fact task so ``create_react_agent``
     substitutes its canned ``_STEP_EXHAUSTED_TEXT`` apology. Because the pre-fix deliverable is
     exactly ``_final_answer(messages)`` — recoverable from the full-capture telemetry — each run
     yields BOTH arms (pre-fix vs post-fix) with no run-to-run variance between them.
  B. Search-backend outage (fix 2). Force ``AgentIO.search`` to return ``None`` and compare the
     model's downstream behavior under the new ``SEARCH BACKEND UNAVAILABLE`` marker against the
     old ``"No results."`` wording (control arm patches the module constant back).
  C. ``always_synthesize`` (fix 4). Run the full task with the opt-in ON; the model's own final
     turn (capture) is what the default OFF path would have returned, so the synthesized answer
     can be scored against it pairwise for rescue-vs-paraphrase-degradation.

Web access is a deterministic in-memory corpus of FICTIONAL entities (a fake ``AgentIO``), so the
probe costs $0, needs no search key, and cannot be answered from parametric memory — only the
LLM is real. Not a pytest test; run manually:

    PYTHONPATH=.:services:agent python3 scripts/langgraph_solver_fix_probe.py \
        [--probe a|b|c|all] [--models qwen2.5:7b,llama3.2:3b] [--replicates 3] \
        [--base-url http://localhost:11435/v1] [--out scripts/_probe_out/langgraph_fixes.json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from langchain_openai import ChatOpenAI  # noqa: E402

from agent.app import langgraph_solver as lgs  # noqa: E402
from agent.app.langgraph_solver import LangGraphSolver, _STEP_EXHAUSTED_TEXT  # noqa: E402
from agent.app.telemetry import TelemetrySession  # noqa: E402

# --------------------------------------------------------------------------------------------
# Fictional corpus. Three facts on three separate pages: the task cannot be finished without at
# least three search+visit rounds, which is what makes the step-exhaustion condition inducible.
# --------------------------------------------------------------------------------------------
MANDATE = (
    "Report three values: (1) the deck span in metres of the Tarnwell Viaduct, (2) the year the "
    "Ostrey Observatory opened, and (3) the population of Kellmoor recorded in the 2021 census. "
    "Give all three values, each with the source URL you read it from."
)

FACTS = {"span": ["412"], "year": ["1897"], "population": ["18,240", "18240"]}

# Second stimulus, aimed squarely at fix 4's target shape: the answer is a DERIVED value, so the
# model has to gather two facts and then do arithmetic — the point at which the review saw small
# models announce the plan ("Now I will compute the difference…") in a tool-call-free turn, which
# LangGraph treats as a natural termination and returns verbatim as the deliverable.
COMPUTE_MANDATE = (
    "Using the web, work out how many years elapsed between the year the Ostrey Observatory "
    "opened and the year the Tarnwell Viaduct was completed. State both years and the final "
    "number of years."
)
COMPUTE_FACTS = {"observatory_year": ["1897"], "viaduct_year": ["1904"],
                 "difference": ["7 years", "7 year", "is 7", "= 7", "elapsed: 7"]}

PAGES: Dict[str, Dict[str, str]] = {
    "https://encyclo.example/tarnwell_viaduct": {
        "title": "Tarnwell Viaduct",
        "keywords": "tarnwell viaduct bridge deck span metres",
        "description": "Railway viaduct in the Tarnwell valley; deck carried on nine piers.",
        "text": (
            "Tarnwell Viaduct\n\nThe Tarnwell Viaduct carries the Marden branch line across the "
            "Tarnwell valley. Completed in 1904 by the engineer Ada Ferrold, the structure has a "
            "deck span of 412 metres and rises 61 metres above the river bed. It was reinforced "
            "in 1972 and is a Grade II listed structure. The viaduct is not to be confused with "
            "the nearby Tarnwell Aqueduct, which is 90 metres long."
        ),
    },
    "https://encyclo.example/ostrey_observatory": {
        "title": "Ostrey Observatory",
        "keywords": "ostrey observatory opened telescope astronomy year",
        "description": "Hilltop observatory on Ostrey Ridge, known for its refractor telescope.",
        "text": (
            "Ostrey Observatory\n\nOstrey Observatory opened in 1897 on Ostrey Ridge, funded by a "
            "bequest from the shipping merchant Halvor Ostrey. Its 24-inch refractor was "
            "installed two years later, in 1899. The observatory was closed between 1941 and "
            "1946 and reopened as a public science centre in 1988."
        ),
    },
    "https://encyclo.example/kellmoor": {
        "title": "Kellmoor",
        "keywords": "kellmoor town population census 2021 inhabitants",
        "description": "Market town on the Kell estuary; population figures by census year.",
        "text": (
            "Kellmoor\n\nKellmoor is a market town on the Kell estuary. The 2021 census recorded "
            "a population of 18,240, up from 16,905 at the 2011 census. The town's economy is "
            "based on boatbuilding and seasonal tourism."
        ),
    },
    "https://encyclo.example/marden_branch_line": {
        "title": "Marden Branch Line",
        "keywords": "marden branch line railway tarnwell viaduct route",
        "description": "Single-track branch line crossing the Tarnwell Viaduct.",
        "text": (
            "Marden Branch Line\n\nThe Marden Branch Line opened in 1906 and runs for 31 "
            "kilometres, crossing the Tarnwell Viaduct at its midpoint. Individual structure "
            "dimensions are listed on their own pages."
        ),
    },
}


class _FakeAgentIO:
    """Stands in for ``AgentIO`` inside ``LangGraphSolver.solve``: same two async methods the
    solver's tools call, backed by the in-memory corpus above. ``dead_search`` reproduces
    ``query_search`` returning ``None`` without raising, which is the exact condition fix 2
    addresses."""

    def __init__(self, *_args, dead_search: bool = False, **_kwargs) -> None:
        self.dead_search = dead_search
        self.searches: List[str] = []
        self.visits: List[str] = []

    async def search(self, query: str, count: int = 10, timeout_seconds: Optional[int] = None):
        self.searches.append(query)
        if self.dead_search:
            return None
        terms = {t for t in ''.join(c.lower() if c.isalnum() else ' ' for c in query).split() if len(t) > 2}
        scored = []
        for url, page in PAGES.items():
            hay = set(page["keywords"].split())
            score = len(terms & hay)
            if score:
                scored.append((score, url, page))
        scored.sort(key=lambda s: -s[0])
        return [{"title": p["title"], "url": u, "description": p["description"]}
                for _s, u, p in scored[:count]]

    async def visit(self, url: str, timeout_seconds: Optional[int] = None):
        self.visits.append(url)
        page = PAGES.get((url or "").strip().rstrip("/"))
        return page["text"] if page else ""


def _capture_turns(telemetry: TelemetrySession) -> List[Dict[str, Any]]:
    """Assistant turns as recorded by ``_record_io_parity(full_capture=True)`` — the same reader a
    real captured benchmark run would use."""
    turns = []
    for event in telemetry.events:
        payload = (event.get("payload") or {}) if isinstance(event, dict) else {}
        inner = payload.get("payload") or {}
        if payload.get("direction") != "out":
            continue
        turns.append({"text": inner.get("completion_text", ""),
                      "tool_calls": inner.get("completion_tool_calls", [])})
    return turns


def _prefix_deliverable(turns: List[Dict[str, Any]]) -> str:
    """What ``LangGraphSolver`` returned BEFORE commit 50f25a34: ``_final_answer(messages)``, i.e.
    the last assistant turn carrying prose rather than a bare tool call."""
    for turn in reversed(turns):
        if turn["tool_calls"]:
            continue
        if (turn["text"] or "").strip():
            return turn["text"]
    return ""


#: Active stimulus, switched by ``--stimulus`` (mandate, fact-checklist).
STIMULUS: Dict[str, Any] = {"mandate": MANDATE, "facts": FACTS}


def _facts_found(text: str) -> List[str]:
    lowered = (text or "").lower()
    return [name for name, variants in STIMULUS["facts"].items()
            if any(v.lower() in lowered for v in variants)]


async def _run_once(model: str, base_url: str, *, max_steps: int, dead_search: bool = False,
                    always_synthesize: bool = False, search_unavailable_text: Optional[str] = None,
                    mandate: Optional[str] = None) -> Dict[str, Any]:
    """One real ``create_react_agent`` run against the fake web and a local ollama model."""
    mandate = mandate or STIMULUS["mandate"]
    io = _FakeAgentIO(dead_search=dead_search)
    telemetry = TelemetrySession(enabled=True, mandate=mandate, correlation_id="probe")

    llm = ChatOpenAI(base_url=base_url, api_key="ollama", model=model, temperature=0.1)
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name=model, full_capture=True, always_synthesize=always_synthesize,
    )
    original_io, original_unavailable = lgs.AgentIO, lgs._SEARCH_UNAVAILABLE
    lgs.AgentIO = lambda **kwargs: io
    LangGraphSolver._build_llm = lambda self: llm  # type: ignore[method-assign]
    if search_unavailable_text is not None:
        lgs._SEARCH_UNAVAILABLE = search_unavailable_text  # control arm: pre-fix wording
    started = time.perf_counter()
    try:
        result = await solver.solve(mandate, max_steps=max_steps, telemetry=telemetry)
    finally:
        lgs.AgentIO, lgs._SEARCH_UNAVAILABLE = original_io, original_unavailable

    turns = _capture_turns(telemetry)
    pre_fix = _prefix_deliverable(turns)
    post_fix = result.get("final_deliverable", "") or ""
    return {
        "model": model,
        "max_steps": max_steps,
        "wall_s": round(time.perf_counter() - started, 1),
        "warning": result.get("warning", ""),
        "assistant_turns": len(turns),
        "searches": io.searches,
        "distinct_searches": len(set(io.searches)),
        "visits": io.visits,
        "tool_calls": [c for t in turns for c in t["tool_calls"]],
        "canned_apology_hit": pre_fix.strip() == _STEP_EXHAUSTED_TEXT,
        "pre_fix_deliverable": pre_fix,
        "pre_fix_facts": _facts_found(pre_fix),
        "post_fix_deliverable": post_fix,
        "post_fix_facts": _facts_found(post_fix),
        "synthesis_ran": bool(post_fix) and post_fix != pre_fix,
    }


async def probe_a(models: List[str], base_url: str, reps: int, max_steps: int) -> List[Dict[str, Any]]:
    """Fix 1: under-budget the step limit until create_react_agent's canned apology appears."""
    rows = []
    for model in models:
        for rep in range(reps):
            row = await _run_once(model, base_url, max_steps=max_steps)
            row.update({"probe": "a_step_exhaustion", "rep": rep})
            rows.append(row)
            print(f"[A] {model} rep{rep}: canned={row['canned_apology_hit']} "
                  f"warn={bool(row['warning'])} pre={row['pre_fix_facts']} post={row['post_fix_facts']}",
                  flush=True)
    return rows


async def probe_b(models: List[str], base_url: str, reps: int, max_steps: int) -> List[Dict[str, Any]]:
    """Fix 2: dead search backend, new marker vs the old "No results." wording."""
    rows = []
    for model in models:
        for rep in range(reps):
            for arm, text in (("new_unavailable", None), ("old_no_results", "No results.")):
                row = await _run_once(model, base_url, max_steps=max_steps, dead_search=True,
                                      search_unavailable_text=text)
                row.update({"probe": "b_dead_search", "arm": arm, "rep": rep})
                rows.append(row)
                print(f"[B] {model} rep{rep} {arm}: searches={len(row['searches'])} "
                      f"distinct={row['distinct_searches']} visits={len(row['visits'])} "
                      f"facts={row['post_fix_facts']}", flush=True)
    return rows


async def probe_c(models: List[str], base_url: str, reps: int, max_steps: int) -> List[Dict[str, Any]]:
    """Fix 4: always_synthesize ON, scored pairwise against the model's own final turn (which is
    exactly what the default-OFF path returns)."""
    rows = []
    for model in models:
        for rep in range(reps):
            row = await _run_once(model, base_url, max_steps=max_steps, always_synthesize=True)
            row.update({"probe": "c_always_synthesize", "rep": rep})
            rows.append(row)
            print(f"[C] {model} rep{rep}: own={row['pre_fix_facts']} synth={row['post_fix_facts']} "
                  f"delta={len(row['post_fix_facts']) - len(row['pre_fix_facts'])}", flush=True)
    return rows


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", default="all", choices=["a", "b", "c", "all"])
    parser.add_argument("--models", default="llama3.2:3b,qwen2.5:7b")
    parser.add_argument("--base-url", default="http://localhost:11435/v1")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--max-steps-exhaust", type=int, default=3)
    parser.add_argument("--max-steps-full", type=int, default=10)
    parser.add_argument("--stimulus", default="report", choices=["report", "compute"])
    parser.add_argument("--out", default="scripts/_probe_out/langgraph_fixes.json")
    args = parser.parse_args()

    if args.stimulus == "compute":
        STIMULUS.update({"mandate": COMPUTE_MANDATE, "facts": COMPUTE_FACTS})

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows: List[Dict[str, Any]] = []
    if args.probe in ("a", "all"):
        rows += await probe_a(models, args.base_url, args.replicates, args.max_steps_exhaust)
    if args.probe in ("b", "all"):
        rows += await probe_b(models, args.base_url, args.replicates, args.max_steps_full)
    if args.probe in ("c", "all"):
        rows += await probe_c(models, args.base_url, args.replicates, args.max_steps_full)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
