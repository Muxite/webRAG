"""Live ($0, local-ollama) probe of commit db8341fe ("attribute every visit result to its source
url and flag repeat visits explicitly", ``agent/app/langgraph_solver.py``).

The unit tests prove the header and the repeat marker are emitted. This probe answers the other
question: under the REAL ``create_react_agent`` loop driven by a real (small, local) model, does
the repeat-visit shape the fix targets actually arise, and when it does, does the marker change
what the model does NEXT?

Stimulus reproduces the live-observed failure (task 065/nano): the model lands on a
DISAMBIGUATION page that names the right article but does not carry the asked-for figure. The
figure lives on a linked sub-article. Pre-fix, the model re-visited the disambiguation URL, got
byte-identical unattributed text back, decided the tool result was "similar to the previous
search result", and answered from parametric memory. The corpus is FICTIONAL, so a
memory-sourced answer is necessarily a fabrication and is detectable as such.

Two paired arms per (model, replicate), differing ONLY in the visit tool's return shape:

  pre   the exact pre-db8341fe return value — bare truncated content, no attribution. Produced by
        stripping the fix's header off the real tool's output, so the two arms cannot drift apart
        for any reason other than the header itself.
  post  the shipped behavior — ``SOURCE: <url>`` on every visit, plus the ``ALREADY VISITED …``
        marker above it on a repeat.

``_SEARCH_VISITED_MARK`` (an uncommitted follow-up in the working tree at the time of writing) is
neutralized in BOTH arms so this probe isolates db8341fe alone; ``--search-mark`` restores it as a
third arm.

Not a pytest test; run manually:

    PYTHONPATH=.:services:agent python3 scripts/langgraph_repeat_visit_probe.py \
        [--models llama3.2:3b,qwen2.5:7b] [--replicates 4] [--max-steps 8] \
        [--base-url http://localhost:11435/v1] [--out scripts/_probe_out/repeat_visit.json]
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from langchain_core.tools import tool  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from agent.app import langgraph_solver as lgs  # noqa: E402
from agent.app.langgraph_solver import LangGraphSolver  # noqa: E402
from agent.app.telemetry import TelemetrySession  # noqa: E402

# --------------------------------------------------------------------------------------------
# Fictional corpus. Only the disambiguation page is in the search index: the sub-articles are
# reachable ONLY by reading their URLs off that page. That models the live shape (a name query
# lands on the disambiguation page) and is what makes a re-visit of the ambiguous page a live
# option for the model instead of an unlikely accident.
# --------------------------------------------------------------------------------------------
MANDATE = (
    "What is the elevation above sea level, in metres, of the village of Riverford in the county "
    "of Calderwych? Give the figure and the source URL you read it from."
)

ANSWER = "214"

DISAMBIG_URL = "https://encyclo.example/riverford"
TARGET_URL = "https://encyclo.example/riverford_calderwych"

PAGES: Dict[str, Dict[str, Any]] = {
    DISAMBIG_URL: {
        "title": "Riverford",
        "indexed": True,
        "keywords": "riverford calderwych village elevation metres sea level county",
        "description": ("Riverford, Calderwych is a village in the county of Calderwych. "
                        "Elevation, population and other figures are in the individual article "
                        "linked from this page."),
        # ``--stimulus linked`` uses this text (each entry carries its URL, so the target page is
        # one read away). ``--stimulus bare`` swaps in DISAMBIG_TEXT_BARE below.
        "text": (
            "Riverford\n\n"
            "Riverford may refer to:\n"
            "  1. Riverford, Calderwych - a village in the county of Calderwych. "
            f"Full article: {TARGET_URL}\n"
            "  2. Riverford Creek - a stream in North Tessany. "
            "Full article: https://encyclo.example/riverford_creek\n"
            "  3. Riverford (surname) - people bearing the surname Riverford. "
            "Full article: https://encyclo.example/riverford_surname\n"
            "  4. Riverford Priory - a ruined monastery near Ambledon. "
            "Full article: https://encyclo.example/riverford_priory\n\n"
            "This is a disambiguation page. It lists articles that share the same title. "
            "Elevation, population and other figures are given in the individual articles "
            "listed above and are not repeated here."
        ),
    },
    TARGET_URL: {
        "title": "Riverford, Calderwych",
        "indexed": False,
        "keywords": "",
        "description": "Village in Calderwych on the eastern flank of Calder Fell.",
        "text": (
            "Riverford, Calderwych\n\n"
            "Riverford is a village in the county of Calderwych, on the eastern flank of Calder "
            "Fell. The village stands at an elevation of 214 metres above sea level. The 2021 "
            "census recorded 1,180 residents. The parish church of St Ferrold dates from 1478."
        ),
    },
    "https://encyclo.example/riverford_creek": {
        "title": "Riverford Creek",
        "indexed": False,
        "keywords": "",
        "description": "Stream in North Tessany.",
        "text": (
            "Riverford Creek\n\nRiverford Creek is a stream in North Tessany. From its source it "
            "falls 38 metres over a course of 11 kilometres before joining the Tessan."
        ),
    },
    "https://encyclo.example/riverford_surname": {
        "title": "Riverford (surname)",
        "indexed": False,
        "keywords": "",
        "description": "People bearing the surname Riverford.",
        "text": (
            "Riverford (surname)\n\nRiverford is a habitational surname. Notable bearers include "
            "Ada Riverford (1822-1901), cartographer, and Halvor Riverford, shipping merchant."
        ),
    },
    "https://encyclo.example/riverford_priory": {
        "title": "Riverford Priory",
        "indexed": False,
        "keywords": "",
        "description": "Ruined monastery near Ambledon.",
        "text": (
            "Riverford Priory\n\nRiverford Priory is a ruined monastery near Ambledon. The ruins "
            "stand at 96 metres above sea level and were surveyed in 1903."
        ),
    },
    # Indexed decoy, so the search index is not degenerate (one hit for every query).
    "https://encyclo.example/calderwych": {
        "title": "Calderwych",
        "indexed": True,
        "keywords": "calderwych county administrative villages calder fell",
        "description": "County in the north-west; contains Calder Fell and several villages.",
        "text": (
            "Calderwych\n\nCalderwych is a county in the north-west. Its highest point is Calder "
            "Fell. Settlement-level statistics are published per village, not in this article."
        ),
    },
}


#: ``--stimulus bare``: the same page as a real cleaned Wikipedia disambiguation page arrives —
#: link TEXT only, no hrefs (``AgentIO.visit`` strips markup). The sub-articles are not in the
#: search index either, so the only moves left are guessing a URL, re-reading this page, or saying
#: the figure is not available. That is the maximum-pressure version of the shape the fix targets.
DISAMBIG_TEXT_BARE = (
    "Riverford\n\n"
    "Riverford may refer to:\n"
    "  Riverford, Calderwych, a village in the county of Calderwych\n"
    "  Riverford Creek, a stream in North Tessany\n"
    "  Riverford (surname), people bearing the surname Riverford\n"
    "  Riverford Priory, a ruined monastery near Ambledon\n\n"
    "This is a disambiguation page. It lists articles that share the same title. Elevation, "
    "population and other figures are given in the individual articles listed above and are not "
    "repeated here."
)


class _FakeAgentIO:
    """Stands in for ``AgentIO`` inside ``LangGraphSolver.solve``: the same two async methods the
    solver's tools call, backed by the in-memory corpus above."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.searches: List[str] = []
        self.visits: List[str] = []

    async def search(self, query: str, count: int = 10, timeout_seconds: Optional[int] = None):
        self.searches.append(query)
        terms = {t for t in ''.join(c.lower() if c.isalnum() else ' ' for c in query).split()
                 if len(t) > 2}
        scored = []
        for url, page in PAGES.items():
            if not page["indexed"]:
                continue
            score = len(terms & set(page["keywords"].split()))
            if score:
                scored.append((score, url, page))
        scored.sort(key=lambda s: -s[0])
        return [{"title": p["title"], "url": u, "description": p["description"]}
                for _s, u, p in scored[:count]]

    async def visit(self, url: str, timeout_seconds: Optional[int] = None):
        self.visits.append(url)
        page = PAGES.get((url or "").strip().rstrip("/"))
        return page["text"] if page else ""


def _make_instrumented_tools(arm: str, log: List[Dict[str, str]]):
    """Wrap the solver's real ``_make_tools``: record every tool result verbatim, and for the
    ``pre`` arm strip the header commit db8341fe added. Stripping (rather than reimplementing the
    tool) guarantees the control arm is byte-identical to the pre-fix return value: the fix
    truncates the same content to the same width and only prepends."""

    real_make_tools = lgs._make_tools  # bound before the module attribute is patched

    def factory(agent_io, search_k, page_chars, retry=None):
        real = {t.name: t for t in real_make_tools(agent_io, search_k, page_chars, retry)}

        @tool
        async def search(query: str) -> str:
            """Web search. Returns titles, URLs, and snippets for the query."""
            out = await real["search"].ainvoke({"query": query})
            log.append({"tool": "search", "arg": query, "result": out})
            return out

        @tool
        async def visit(url: str) -> str:
            """Fetch a page's text content. Use an exact URL from search results."""
            out = await real["visit"].ainvoke({"url": url})
            if arm == "source_only":
                # Half of the fix: keep the SOURCE line, drop the repeat marker, to tell the two
                # halves apart when the paired pre/post arms differ.
                out = out.removeprefix(f"{lgs._VISIT_REPEAT}\n")
            if arm == "pre":
                for header in (f"{lgs._VISIT_REPEAT}\n{lgs._VISIT_SOURCE_PREFIX}{url}\n",
                               f"{lgs._VISIT_SOURCE_PREFIX}{url}\n"):
                    if out.startswith(header):
                        out = out[len(header):]
                        break
            log.append({"tool": "visit", "arg": url, "result": out})
            return out

        return [search, visit]

    return factory


def _capture_turns(telemetry: TelemetrySession) -> List[Dict[str, Any]]:
    turns = []
    for event in telemetry.events:
        payload = (event.get("payload") or {}) if isinstance(event, dict) else {}
        inner = payload.get("payload") or {}
        if payload.get("direction") != "out":
            continue
        turns.append({"text": inner.get("completion_text", ""),
                      "tool_calls": inner.get("completion_tool_calls", [])})
    return turns


_ELEVATION_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:m\b|metre|meter)", re.IGNORECASE)
_ABSTAIN_RE = re.compile(
    r"not (?:available|provided|found|specified|given|stated|listed|determined)"
    r"|cannot be determined|could not (?:find|be found)|couldn't find|no (?:elevation|figure)"
    r"|does not (?:contain|provide|include|give|specify|state)|unable to (?:find|determine)"
    r"|not (?:mentioned|present) (?:on|in)",
    re.IGNORECASE)


def _score(text: str) -> Dict[str, Any]:
    """Correct / fabricated / abstained, over the FICTIONAL corpus: any elevation figure other
    than 214 cannot have come from a page and is therefore invented or mis-attributed."""
    numbers = [n.replace(",", "") for n in _ELEVATION_RE.findall(text or "")]
    correct = ANSWER in numbers or ANSWER in (text or "")
    wrong = [n for n in numbers if n != ANSWER and n not in ("1180", "1,180")]
    return {"correct": correct, "wrong_numbers": wrong,
            "fabricated": bool(wrong) and not correct,
            "abstained": bool(_ABSTAIN_RE.search(text or "")) and not correct}


async def _run_once(model: str, base_url: str, arm: str, *, max_steps: int,
                    search_mark: bool, temperature: float) -> Dict[str, Any]:
    io = _FakeAgentIO()
    log: List[Dict[str, str]] = []
    telemetry = TelemetrySession(enabled=True, mandate=MANDATE, correlation_id="probe")
    llm = ChatOpenAI(base_url=base_url, api_key="ollama", model=model, temperature=temperature)
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name=model, full_capture=True,
    )
    originals = (lgs.AgentIO, lgs._make_tools, lgs._SEARCH_VISITED_MARK)
    lgs.AgentIO = lambda **kwargs: io
    lgs._make_tools = _make_instrumented_tools(arm, log)
    if not search_mark:
        lgs._SEARCH_VISITED_MARK = ""
    LangGraphSolver._build_llm = lambda self: llm  # type: ignore[method-assign]
    started = time.perf_counter()
    error = ""
    try:
        result = await solver.solve(MANDATE, max_steps=max_steps, telemetry=telemetry)
    except Exception as exc:  # a model with no tool-calling endpoint fails here
        result, error = {}, f"{type(exc).__name__}: {exc}"
    finally:
        lgs.AgentIO, lgs._make_tools, lgs._SEARCH_VISITED_MARK = originals

    turns = _capture_turns(telemetry)
    final = result.get("final_deliverable", "") or ""
    visits = [v["arg"] for v in log if v["tool"] == "visit"]
    repeats = sorted({u for u in visits if visits.count(u) > 1})
    # Index of the first repeated visit in the tool log, so the turns AFTER it can be read.
    repeat_idx, seen = None, set()
    for i, entry in enumerate(log):
        if entry["tool"] != "visit":
            continue
        if entry["arg"] in seen:
            repeat_idx = i
            break
        seen.add(entry["arg"])
    repeat_entry = log[repeat_idx] if repeat_idx is not None else None
    return {
        "model": model, "arm": arm, "max_steps": max_steps, "temperature": temperature,
        "error": error,
        "wall_s": round(time.perf_counter() - started, 1),
        "warning": result.get("warning", ""),
        "searches": io.searches,
        "visits": visits,
        "repeat_visit": bool(repeats),
        "repeat_urls": repeats,
        "visited_target": TARGET_URL in visits,
        "visited_disambig": DISAMBIG_URL in visits,
        "repeat_result_head": (repeat_entry["result"][:220] if repeat_entry else ""),
        "repeat_marker_delivered": bool(repeat_entry
                                        and repeat_entry["result"].startswith(lgs._VISIT_REPEAT)),
        "turns_after_repeat": [t["text"] for t in turns][-4:] if repeat_entry else [],
        "final": final,
        **_score(final),
        "tool_log": log,
        "turns": turns,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="llama3.2:3b,qwen2.5:7b")
    parser.add_argument("--arms", default="pre,post")
    parser.add_argument("--base-url", default="http://localhost:11435/v1")
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--stimulus", default="linked", choices=["linked", "bare"],
                        help="linked: the disambiguation page prints each sub-article's URL. "
                             "bare: link text only (real cleaned-Wikipedia shape), target page "
                             "unindexed and unlinked.")
    parser.add_argument("--search-mark", action="store_true",
                        help="leave _SEARCH_VISITED_MARK active (uncommitted follow-up) instead of "
                             "neutralizing it; isolates db8341fe when off")
    parser.add_argument("--out", default="scripts/_probe_out/repeat_visit.json")
    args = parser.parse_args()

    if args.stimulus == "bare":
        PAGES[DISAMBIG_URL]["text"] = DISAMBIG_TEXT_BARE

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    rows: List[Dict[str, Any]] = []
    for model in models:
        for rep in range(args.replicates):
            for arm in arms:
                row = await _run_once(model, args.base_url, arm, max_steps=args.max_steps,
                                      search_mark=args.search_mark,
                                      temperature=args.temperature)
                row["rep"], row["stimulus"] = rep, args.stimulus
                rows.append(row)
                print(f"[{model} rep{rep} {arm}] visits={len(row['visits'])} "
                      f"repeat={row['repeat_visit']} marker={row['repeat_marker_delivered']} "
                      f"target={row['visited_target']} correct={row['correct']} "
                      f"fabricated={row['fabricated']} abstain={row['abstained']} "
                      f"{row['error']}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
