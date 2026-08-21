"""Live ($0, local-ollama) validation of the four deterministic mechanisms shipped on
``strong-agent-trace-guidance`` (commits ``b94cb026`` and ``7d080088``).

All four were already validated offline (unit tests) and by REPLAYING historical captures. What
neither of those can answer is the question this probe exists for: with the flag actually toggled
ON and a FRESH completion coming back from a real model, does the mechanism fire where it should,
stay quiet where it should, and change the verdict for the better?

Every stimulus is real. Page text is either lifted verbatim out of ``agent/idea_test_results/``
captures or fetched once from Wikipedia into ``scripts/_probe_fixtures/`` (no LLM involved in
either case); the only synthetic element anywhere is the numeric INJECTION arm, which perturbs
the model's own real answer text and is labelled as such.

  N1  numeric provenance / real      task 150, merge ``b71fb7b4`` of the llama-3.2-3b
                                     ``dag_v2_fixes_revalidation`` run: two real Hardanger Bridge
                                     visits stating 1,310 m. A flag firing here is a FALSE
                                     POSITIVE -- this arm exists to measure that.
  N2  numeric provenance / decoy     task 150, merge ``b22b3639`` of the nano
                                     ``dag_v2_playbook_race`` run -- the run that produced the
                                     "all three routes confirm 575 meters" hallucination. Its
                                     real evidence is two BROOKLYN Bridge pages under the
                                     Hardanger mandate, so any Hardanger figure the model states
                                     is unsourced. Live hallucination induction.
  R1  race value agreement / agree   two real captured Hardanger visits (en.wikipedia +
                                     no.wikipedia 'Hardangerbrua') registered as one race group.
  R2  race value agreement / disagree the en.wikipedia Hardanger visit against the real
                                     100k-char Brooklyn Bridge page the nano run actually opened
                                     for the same mandate.
  C1  candidate roster / four pages  task 122, merge ``e183df66``: all four telescope pages
                                     really opened. The roster should come back complete-ish.
  C2  candidate roster / two pages   task 122, merge ``34812b4e``: one Arecibo visit and a
                                     Serper-403-shaped empty search. Three candidates have no
                                     page at all.
  H1  chain closure / closed         task 065: the real captured Pablo Neruda page followed by
                                     the real captured 'Parral, Chile' page, whose body reads
                                     "Parral is the birthplace of poet Pablo Neruda" (elevation
                                     162 m).
  H2  chain closure / drifted        the same Neruda page followed by the real Temuco article
                                     (elevation 360 m), which names Neruda only as having LIVED
                                     there -- the chain-poisoning decoy the strong-agent trace
                                     refused, reproduced from the live page rather than a
                                     hand-built reconstruction.

Arms are PAIRED, which is what makes 4 replicates enough against this session's known run-to-run
variance: the flag under test provably never touches prompt assembly, so each cell makes ONE real
completion with the flag OFF and then re-runs the whole ``MergeLeafAction.execute`` pipeline with
the flag ON against a replayed identical response. The prompts of the two runs are compared byte
for byte and any mismatch is recorded rather than swallowed.

The wire is the engine's own path (a real ``ConnectorLLM`` with ``LLM_PROVIDER=ollama`` +
``LLM_NUM_CTX``, i.e. ``OllamaNativeBackend``), so ``num_ctx`` is honoured instead of being
silently dropped by the OpenAI shim; only the model endpoint is local.

Not a pytest test; run manually:

    PYTHONPATH=.:services:agent python3 scripts/trace_mechanisms_local_probe.py \
        [--models qwen2.5:7b,llama3.2:3b] [--replicates 4] [--num-ctx 32768] \
        [--host http://localhost:11435] [--out scripts/_probe_out/trace_mechanisms.json]
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "services"))
sys.path.insert(0, str(REPO / "agent"))

RESULTS = REPO / "agent" / "idea_test_results"
FIXTURES = REPO / "scripts" / "_probe_fixtures"

# --- captured stimulus ---------------------------------------------------------------------

_CAP_150_REAL = ("dag_v2_fixes_revalidation_20260821_150_meta-llama-llama-3.2-3b-instruct"
                 "_graph_cfg28dc3825_r1.json", "b71fb7b4")
_CAP_150_DECOY = ("dag_v2_playbook_race_20260821_150_openai-gpt-4.1-nano"
                  "_graph_cfgcb42ed79_r1.json", "b22b3639")
_CAP_122_FOUR = ("dag_v2_playbook_race_20260821_122_meta-llama-llama-3.2-3b-instruct"
                 "_graph_cfgcb42ed79_r1.json", "e183df66")
_CAP_122_TWO = ("dag_v2_playbook_race_20260821_122_openai-gpt-4.1-nano"
                "_graph_cfg370b23e7_r1.json", "34812b4e")
_CAP_065_NERUDA = ("capability_spectrum_v2_20260820_065_openai-gpt-4.1-nano"
                   "_graph_cfg690b88ed_r1.json", "https://en.wikipedia.org/wiki/Pablo_Neruda")
_CAP_065_PARRAL = ("capability_spectrum_v2_20260820_065_meta-llama-llama-3.2-3b-instruct"
                   "_graph_cfg690b88ed_r1.json", "https://en.wikipedia.org/wiki/Parral,_Chile")
_CAP_150_NO = ("dag_v2_fixes_revalidation_20260821_150_meta-llama-llama-3.2-3b-instruct"
               "_graph_cfg28dc3825_r1.json", "https://no.wikipedia.org/wiki/Hardangerbrua")
_CAP_150_EN = ("dag_v2_fixes_revalidation_20260821_150_meta-llama-llama-3.2-3b-instruct"
               "_graph_cfg28dc3825_r1.json", "https://en.wikipedia.org/wiki/Hardanger_Bridge")
_CAP_150_BROOKLYN = ("dag_v2_playbook_race_20260821_150_openai-gpt-4.1-nano"
                     "_graph_cfgcb42ed79_r1.json", "https://en.wikipedia.org/wiki/Brooklyn_Bridge")

#: Articles fetched once into ``_probe_fixtures`` (see ``--refresh-fixtures``), because no
#: capture in this repo stores their raw bodies: nothing ever went to the Temuco decoy (the
#: offline test had to hand-build that page), and no captured task-150 run ever reached ROUTE 2.
_FETCHED = {
    "Temuco": "Temuco",
    "List_of_longest_suspension_bridge_spans": "List of longest suspension bridge spans",
}


def _nodes(filename: str) -> Dict[str, Any]:
    return json.loads((RESULTS / filename).read_text())["execution"]["graph"]["nodes"]


def _merge_details(spec: Tuple[str, str]) -> Dict[str, Any]:
    filename, prefix = spec
    for nid, node in _nodes(filename).items():
        details = node.get("details") or {}
        if nid.startswith(prefix) and details.get("action") == "merge":
            return details
    raise SystemExit(f"merge node {prefix} not found in {filename}")


def _visit_result(spec: Tuple[str, str]) -> Dict[str, Any]:
    filename, url = spec
    for node in _nodes(filename).values():
        ar = (node.get("details") or {}).get("action_result") or {}
        if isinstance(ar, dict) and ar.get("action") == "visit" and ar.get("success") \
                and ar.get("url") == url:
            return copy.deepcopy(ar)
    raise SystemExit(f"visit {url} not found in {filename}")


def _wiki_result(slug: str, refresh: bool) -> Dict[str, Any]:
    """A real Wikipedia article as a ``visit`` payload, cached under ``_probe_fixtures``.

    Same shape ``VisitLeafAction`` stores, so the mechanisms cannot tell it from a captured
    visit. Fetching costs nothing and involves no model.
    """
    title = _FETCHED[slug]
    path = FIXTURES / f"{slug.lower()}.json"
    if refresh or not path.exists():
        import httpx
        from bs4 import BeautifulSoup

        # Wikimedia's REST API answers a default client library UA with a 126-character
        # robot-policy notice instead of the article, which is silently indistinguishable from
        # a very short page; the explicit UA is what makes the fixture the real article.
        html = httpx.get(
            f"https://en.wikipedia.org/api/rest_v1/page/html/{slug}",
            timeout=60.0, follow_redirects=True,
            headers={"User-Agent": "webRAG-offline-probe/1.0 (local mechanism validation)"},
        ).text
        if len(html) < 5000:
            raise SystemExit(f"fetch for {slug} returned {len(html)} chars: {html[:200]!r}")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = re.sub(r"\n{2,}", "\n", soup.get_text("\n"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
    text = json.loads(path.read_text())["text"]
    return {
        "action": "visit", "success": True,
        "url": f"https://en.wikipedia.org/wiki/{slug}",
        "page_title": f"{title} - Wikipedia", "h1_text": title,
        "content_full": text, "content": text[:8000],
    }


def _entry(title: str, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"title": title, "result": result, "is_merge": False}


# --- graph assembly ------------------------------------------------------------------------


def _build_graph(mandate: str, merge_goal: str, entries: List[Dict[str, Any]],
                 race_members: Optional[List[int]] = None):
    """root(mandate) -> parent -> [one node per merged entry] + the merge node itself.

    The children are real nodes carrying their captured ``action_result``, because every one of
    the four mechanisms reads fetched page text off the GRAPH rather than off the merge payload.
    """
    from agent.app.idea_dag import IdeaDag
    from agent.app.idea_policies.alternative_branch import RACE_GROUPS
    from agent.app.idea_policies.base import DetailKey, IdeaNodeStatus

    graph = IdeaDag(root_title="root")
    root = graph.get_node(graph.root_id())
    root.details[DetailKey.GOAL.value] = mandate
    root.details[DetailKey.ORIGINAL_GOAL.value] = mandate
    root.details["mandate"] = mandate

    parent = graph.add_child(graph.root_id(), "parent", status=IdeaNodeStatus.ACTIVE)
    child_ids: List[str] = []
    for entry in entries:
        child = graph.add_child(parent.node_id, entry.get("title") or "child",
                                status=IdeaNodeStatus.DONE)
        result = entry.get("result") or {}
        child.details[DetailKey.ACTION_RESULT.value] = result
        if isinstance(result, dict) and result.get("action"):
            child.details[DetailKey.ACTION.value] = result["action"]
        child_ids.append(child.node_id)

    merge = graph.add_child(parent.node_id, "merge", status=IdeaNodeStatus.PENDING)
    merge.details[DetailKey.MERGED_RESULTS.value] = entries
    merge.details[DetailKey.GOAL.value] = merge_goal
    merge.details[DetailKey.ORIGINAL_GOAL.value] = merge_goal
    if race_members:
        parent.details[RACE_GROUPS] = {
            "probe-route-race": [child_ids[i] for i in race_members]
        }
    return graph, merge


# --- scenarios ----------------------------------------------------------------------------


def _scenarios(refresh: bool) -> Dict[str, Dict[str, Any]]:
    from agent.app.idea_tests import test_065_tier5_leak_resistant_chain as t065
    from agent.app.idea_tests import test_122_tier5_filled_aperture_telescope_survivor as t122
    from agent.app.idea_tests import test_150_tier5_race_merge_bridge_span as t150

    m065, m122, m150 = (t065.get_task_statement(), t122.get_task_statement(),
                        t150.get_task_statement())

    d150_real, d150_decoy = _merge_details(_CAP_150_REAL), _merge_details(_CAP_150_DECOY)
    d122_four, d122_two = _merge_details(_CAP_122_FOUR), _merge_details(_CAP_122_TWO)

    neruda = _visit_result(_CAP_065_NERUDA)
    parral = _visit_result(_CAP_065_PARRAL)
    temuco = _wiki_result("Temuco", refresh)
    span_list = _wiki_result("List_of_longest_suspension_bridge_spans", refresh)
    hard_en, hard_no = _visit_result(_CAP_150_EN), _visit_result(_CAP_150_NO)
    brooklyn = _visit_result(_CAP_150_BROOKLYN)

    return {
        "N1_numeric_real": {
            "mechanism": "numeric_provenance", "flag": "merge_require_numeric_provenance_enabled",
            "expect": "quiet",
            "mandate": m150, "merge_goal": m150,
            "entries": d150_real["merged_results"],
        },
        "N2_numeric_decoy": {
            "mechanism": "numeric_provenance", "flag": "merge_require_numeric_provenance_enabled",
            "expect": "fire",
            "mandate": m150, "merge_goal": m150,
            "entries": d150_decoy["merged_results"],
        },
        "R1_race_agree": {
            "mechanism": "race_value_agreement", "flag": "merge_race_value_agreement_enabled",
            "expect": "quiet",
            "mandate": m150, "merge_goal": m150,
            "entries": [_entry("ROUTE 1 - English Wikipedia Hardanger Bridge", hard_en),
                        _entry("ROUTE 3 - Norwegian Wikipedia Hardangerbrua", hard_no)],
            "race_members": [0, 1],
        },
        "R2_race_disagree": {
            "mechanism": "race_value_agreement", "flag": "merge_race_value_agreement_enabled",
            "expect": "fire",
            "mandate": m150, "merge_goal": m150,
            "entries": [_entry("ROUTE 1 - English Wikipedia Hardanger Bridge", hard_en),
                        _entry("ROUTE 2 - ranked list of longest suspension spans", brooklyn)],
            "race_members": [0, 1],
        },
        # The false-positive arm the other two cannot supply: ROUTE 1 and ROUTE 2 exactly as the
        # mandate describes them, both stating the same 1,310 m main span -- but ROUTE 1's page
        # also prints the 1,380 m TOTAL length one line above it, which is the trap the mandate
        # itself warns about. A ``disagree`` here would downgrade a correct, genuinely
        # cross-confirmed answer.
        "R3_race_agree_two_routes": {
            "mechanism": "race_value_agreement", "flag": "merge_race_value_agreement_enabled",
            "expect": "quiet",
            "mandate": m150, "merge_goal": m150,
            "entries": [_entry("ROUTE 1 - English Wikipedia Hardanger Bridge", hard_en),
                        _entry("ROUTE 2 - ranked list of longest suspension spans", span_list)],
            "race_members": [0, 1],
        },
        "C1_roster_four": {
            "mechanism": "candidate_roster", "flag": "merge_candidate_roster_enabled",
            "expect": "quiet",
            "mandate": m122, "merge_goal": m122,
            "entries": d122_four["merged_results"],
        },
        "C2_roster_two": {
            "mechanism": "candidate_roster", "flag": "merge_candidate_roster_enabled",
            "expect": "fire",
            "mandate": m122, "merge_goal": m122,
            "entries": d122_two["merged_results"],
        },
        "H1_chain_closed": {
            "mechanism": "chain_closure", "flag": "merge_chain_closure_enabled",
            "expect": "quiet",
            "mandate": m065, "merge_goal": m065,
            "entries": [_entry("Read Pablo Neruda's birthplace", neruda),
                        _entry("Read the birth town's elevation", parral)],
        },
        "H2_chain_drifted": {
            "mechanism": "chain_closure", "flag": "merge_chain_closure_enabled",
            "expect": "fire",
            "mandate": m065, "merge_goal": m065,
            "entries": [_entry("Read Pablo Neruda's birthplace", neruda),
                        _entry("Read the birth town's elevation", temuco)],
        },
    }


# --- deterministic audits, computed independently of any completion -------------------------


def _static_audit(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """What the mechanisms say about the STIMULUS alone (no model output involved).

    Reported alongside the live cells so a mechanism that never fires can be told apart from a
    mechanism whose stimulus never reached it.
    """
    from agent.app.idea_policies.candidate_roster import audit_candidate_roster
    from agent.app.idea_policies.chain_closure import audit_chain_closure
    from agent.app.idea_policies.merge import race_group_value_verdicts

    graph, merge = _build_graph(scenario["mandate"], scenario["merge_goal"],
                                scenario["entries"], scenario.get("race_members"))
    parent = graph.get_node(merge.parent_id)
    closure = audit_chain_closure(graph)
    roster = audit_candidate_roster(graph, scenario["mandate"], "", scenario["entries"])
    return {
        "race_verdicts": race_group_value_verdicts(graph, parent),
        "chain_closure": closure.as_dict(),
        "roster": roster.as_registry(),
        "roster_active": roster.active,
        "roster_gaps": roster.missing_requirements(),
    }


# --- live plumbing --------------------------------------------------------------------------


def _configure_env(host: str, model: str, num_ctx: int, timeout: float) -> None:
    if not re.match(r"^https?://(localhost|127\.0\.0\.1)(:\d+)?/$", host.rstrip("/") + "/"):
        raise SystemExit(f"refusing non-local endpoint {host!r}: this probe is $0/local only")
    os.environ["LLM_PROVIDER"] = "ollama"
    os.environ["MODEL_API_URL"] = host.rstrip("/") + "/v1"
    os.environ["LLM_API_KEY"] = "ollama"
    os.environ["MODEL_NAME"] = model
    os.environ["LLM_NUM_CTX"] = str(num_ctx)
    os.environ["LLM_READ_TIMEOUT"] = str(int(timeout))
    os.environ["LLM_CONNECT_TIMEOUT"] = "10"


class _LiveIO:
    """The real ``AgentIO`` over a real ``ConnectorLLM``, taping prompt and completion."""

    def __init__(self):
        from agent.app.agent_io import AgentIO
        from agent.app.connector_llm import ConnectorLLM
        from shared.connector_config import ConnectorConfig

        self._io = AgentIO(connector_llm=ConnectorLLM(ConnectorConfig()), connector_search=None,
                           connector_http=None, connector_chroma=None)
        self.connector_llm = self._io.connector_llm
        self.messages: List[Dict[str, str]] = []
        self.response = ""

    def build_llm_payload(self, **kwargs) -> Dict[str, Any]:
        self.messages = kwargs.get("messages") or []
        return self._io.build_llm_payload(**kwargs)

    async def query_llm_with_fallback(self, payload, **kwargs) -> str:
        self.response = await self._io.query_llm_with_fallback(payload, **kwargs) or ""
        return self.response


class _ReplayIO:
    """Same interface, returning a completion recorded by :class:`_LiveIO`.

    The flag under test is read only AFTER the completion comes back, so replaying it is what
    turns the ON/OFF comparison into a paired one instead of two draws from a noisy model.
    ``messages`` is still recorded, and the caller asserts it matches the live run's byte for
    byte -- that assertion is the evidence the arms really are prompt-identical.
    """

    def __init__(self, response: str):
        self.response = response
        self.messages: List[Dict[str, str]] = []

    def build_llm_payload(self, **kwargs) -> Dict[str, Any]:
        self.messages = kwargs.get("messages") or []
        return {"messages": self.messages}

    async def query_llm_with_fallback(self, payload, **kwargs) -> str:
        return self.response


_INJECT_VALUE = "8 271"


def _inject_numbers(text: str) -> str:
    """The model's own answer with every measurement replaced by one no page can contain.

    The fallback the brief allows when a live hallucination does not reproduce: everything else
    in the pipeline (graph, evidence, claim wording) stays exactly as the model produced it, so
    what is measured is the check's sensitivity rather than a hand-written sentence.
    """
    from agent.app.idea_policies.grounding import _CLAIM_RE

    return _CLAIM_RE.sub(lambda m: f"{_INJECT_VALUE} {m.group(2)}", str(text or ""))


async def _run_cell(scenario_id: str, scenario: Dict[str, Any], flag_on: bool, model: str,
                    args, io) -> Dict[str, Any]:
    from agent.app.idea_dag_settings import load_idea_dag_settings
    from agent.app.idea_policies.actions import MergeLeafAction
    from agent.app.idea_policies.candidate_roster import (
        CANDIDATE_ROSTER, ROSTER_MAGNITUDE_MARKER, ROSTER_UNDISPOSED_MARKER,
    )
    from agent.app.idea_policies.chain_closure import CHAIN_CLOSURE, CHAIN_CLOSURE_OPEN_MARKER
    from agent.app.idea_policies.grounding import answer_numeric_provenance

    graph, node = _build_graph(scenario["mandate"], scenario["merge_goal"],
                               scenario["entries"], scenario.get("race_members"))
    settings = load_idea_dag_settings()
    settings[scenario["flag"]] = flag_on
    settings["merge_model"] = model
    settings["llm_timeout_seconds"] = args.timeout

    started = time.perf_counter()
    error = ""
    try:
        result = await MergeLeafAction(settings=settings).execute(graph, node.node_id, io)
    except Exception as exc:  # noqa: BLE001
        result, error = {}, f"{type(exc).__name__}: {exc}"

    synthesized = (result or {}).get("synthesized") or {}
    claim = MergeLeafAction._claim_text(synthesized)
    numeric = answer_numeric_provenance(graph, claim, scenario["entries"])
    injected = answer_numeric_provenance(graph, _inject_numbers(claim), scenario["entries"])
    root = graph.get_node(graph.root_id())
    return {
        "scenario": scenario_id, "mechanism": scenario["mechanism"],
        "expect": scenario["expect"], "arm": "on" if flag_on else "off", "model": model,
        "error": error, "wall_s": round(time.perf_counter() - started, 1),
        "prompt_chars": sum(len(m.get("content", "")) for m in io.messages),
        "goal_achieved": (result or {}).get("goal_achieved"),
        "raw_goal_achieved": synthesized.get("goal_achieved") if isinstance(synthesized, dict) else None,
        "missing_requirements": (result or {}).get("missing_requirements") or [],
        "goal_evaluation": ((result or {}).get("goal_evaluation") or "")[:600],
        "summary": (synthesized.get("summary") if isinstance(synthesized, dict) else "") or "",
        "claim_text": claim[:1500],
        # --- mechanism outputs, as stamped on the graph by the real pipeline ---
        "numeric_unverified": node.details.get("goal_achieved_numeric_unverified"),
        "numeric_verified_values": [c.raw for c in numeric.verified],
        "numeric_all_unverified": [c.raw for c in numeric.unverified],
        "numeric_unsupported": numeric.unsupported,
        "numeric_checked": numeric.checked,
        "injected_unsupported": injected.unsupported,
        "injected_checked": injected.checked,
        "race_value_disagreement": node.details.get("race_value_disagreement"),
        "roster": (root.details.get(CANDIDATE_ROSTER) if isinstance(root.details, dict) else None),
        "roster_undisposed": node.details.get(ROSTER_UNDISPOSED_MARKER),
        "roster_magnitude_tripwire": node.details.get(ROSTER_MAGNITUDE_MARKER),
        "chain_closure": node.details.get(CHAIN_CLOSURE),
        "chain_closure_open": node.details.get(CHAIN_CLOSURE_OPEN_MARKER),
        "merge_should_skip": bool(node.details.get("merge_should_skip")),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="qwen2.5:7b,llama3.2:3b")
    parser.add_argument("--scenarios", default="")
    parser.add_argument("--host", default="http://localhost:11435")
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--num-ctx", type=int, default=32768)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--refresh-fixtures", action="store_true")
    parser.add_argument("--static-only", action="store_true",
                        help="print the stimulus-only audits and exit (no model calls)")
    parser.add_argument("--out", default="scripts/_probe_out/trace_mechanisms.json")
    args = parser.parse_args()

    scenarios = _scenarios(args.refresh_fixtures)
    wanted = [s.strip() for s in args.scenarios.split(",") if s.strip()] or list(scenarios)

    statics = {sid: _static_audit(scenarios[sid]) for sid in wanted}
    for sid in wanted:
        st = statics[sid]
        print(f"[static {sid}] race={st['race_verdicts']} "
              f"chain_active={st['chain_closure']['active']}/closed={st['chain_closure']['closed']} "
              f"roster_active={st['roster_active']} gaps={len(st['roster_gaps'])}", flush=True)
    if args.static_only:
        return

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows: List[Dict[str, Any]] = []
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for model in models:
        _configure_env(args.host, model, args.num_ctx, args.timeout)
        live = _LiveIO()
        for rep in range(args.replicates):
            for sid in wanted:
                scenario = scenarios[sid]
                off = await _run_cell(sid, scenario, False, model, args, live)
                replay = _ReplayIO(live.response)
                on = await _run_cell(sid, scenario, True, model, args, replay)
                on["prompt_identical"] = replay.messages == live.messages
                for row in (off, on):
                    row["rep"] = rep
                    row["static"] = statics[sid]
                    rows.append(row)
                print(f"[{model} rep{rep} {sid}] expect={scenario['expect']} "
                      f"off:goal={off['goal_achieved']} on:goal={on['goal_achieved']} "
                      f"num_unsup={on['numeric_unsupported']} inj={on['injected_unsupported']} "
                      f"race={on['race_value_disagreement']} "
                      f"roster_gap={on['roster_undisposed']} "
                      f"chain_open={bool(on['chain_closure_open'])} "
                      f"same_prompt={on['prompt_identical']} {off['wall_s']}s {off['error']}",
                      flush=True)
                out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
