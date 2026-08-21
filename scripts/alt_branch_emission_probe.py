"""One-off, offline-runnable diagnostic for the 2026-08-21 alternative_of/race_group prompt fix
(see docs/handoffs/DAG_V2_REASONING_DIFF_FINDINGS_2026-08-21.md, item 3, and
agent/app/idea_policies/expansion.py's ``_ALTERNATIVE_BRANCH_ADDENDUM``).

Builds the REAL expansion system prompt via ``LlmExpansionPolicy._build_messages`` (same code
path the engine uses), swaps in the OLD (pre-fix) addendum text for a control run, and sends
both variants to a local ollama model across a handful of stimulus mandates adapted from tasks
122 and 150 (both plausibly trigger multi-approach/race/fallback reasoning). Counts how often
``alternative_of``/``race_group`` appear in the raw completion text, old addendum vs. new.

Not a pytest test — a standalone diagnostic script, run manually:
    PYTHONPATH=.:services:agent python3 scripts/alt_branch_emission_probe.py [--model NAME] [--base-url URL]

$0 cost: local ollama inference only, no OpenRouter/live-API calls.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from agent.app.idea_dag import IdeaDag  # noqa: E402
from agent.app.idea_policies.expansion import (  # noqa: E402
    LlmExpansionPolicy,
    _ALTERNATIVE_BRANCH_ADDENDUM as NEW_ADDENDUM,
)
from agent.app.llm_backends import LLMBackend  # noqa: E402

# The pre-fix addendum text, verbatim from commit da8a8513 (agent/app/idea_policies/expansion.py
# lines 186-195 at that commit). Kept inline here rather than re-fetched from git so the probe
# stays runnable standalone.
OLD_ADDENDUM = (
    "MULTIPLE APPROACHES: when a sub-problem has more than one candidate route, say so with "
    "one of two OPTIONAL top-level fields on a candidate. Use \"alternative_of\": "
    "\"<exact title of another candidate>\" when this candidate is a BACKUP that should only "
    "run if that other candidate does not work out — the backup is held back until then. Use "
    "\"race_group\": \"<short label>\" on TWO OR MORE candidates that are different ways to "
    "find the SAME fact and are worth trying at once; put the identical label on every member "
    "of the group, and use a different label (or none) for candidates that find a DIFFERENT "
    "fact. Most candidates need neither field. Omit both when in doubt."
)

# Stimulus mandates: adapted from task 150's actual race-and-merge mandate (three independent
# redundant routes to one value — the canonical race_group shape) and task 122's actual
# elimination mandate (four candidates, one survivor after checking each — a plausible
# alternative_of/fallback shape), plus one hand-adapted variant of each to broaden the sample
# without needing full task-module machinery.
STIMULI: List[Dict[str, str]] = [
    {
        "id": "150_race_bridge_span",
        "mandate": (
            "You are given NO URLs — navigate the web yourself and READ the pages (do not guess "
            "from memory). ONE value is wanted: the length of the MAIN SPAN, in metres, of the "
            "Hardanger Bridge in Norway. Three INDEPENDENT routes each state that value in full: "
            "(1) the bridge's own English Wikipedia article; (2) English Wikipedia's ranked list "
            "of the longest suspension bridge spans; (3) the Norwegian Wikipedia article "
            "'Hardangerbrua'. They do not depend on one another, and ANY ONE OF THEM IS "
            "SUFFICIENT. Pursue the routes CONCURRENTLY rather than one after another, and answer "
            "as soon as a route delivers the value; if a route stalls, drop it and answer from "
            "another. Report the main span in metres and the exact source URL of every page you "
            "read."
        ),
    },
    {
        "id": "122_survivor_telescope",
        "mandate": (
            "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess "
            "from memory). Four large single-dish radio telescopes are candidates: Arecibo "
            "Observatory, FAST (Five-hundred-meter Aperture Spherical Telescope), Green Bank "
            "Telescope, and RATAN-600. Exactly ONE of these is the largest FILLED-APERTURE "
            "single-dish radio telescope CURRENTLY IN OPERATION — one has collapsed and is no "
            "longer operational, one is a ring reflector, one is a smaller fully steerable dish. "
            "Open EACH telescope's page and check its status before committing to an answer; do "
            "NOT simply pick the most famous name. Report which telescope survives and its "
            "illuminated aperture diameter in metres, citing the exact Wikipedia URL of every "
            "page you read."
        ),
    },
    {
        "id": "adapted_race_population",
        "mandate": (
            "You are given NO URLs. ONE value is wanted: the current population of Reykjavik, "
            "Iceland. At least two independent sources plausibly state this figure: the city's own "
            "official statistics page, and its English Wikipedia infobox. Either source alone is "
            "sufficient to answer; you do not need both. If one source is slow or unreachable, use "
            "the other instead. Report the population and the exact source URL you used."
        ),
    },
    {
        "id": "adapted_fallback_founding_year",
        "mandate": (
            "You are given NO URLs. Find the exact founding year of the University of Bologna. "
            "Your first choice should be to check the university's own official history page; if "
            "that page cannot be reached or does not state a year, fall back to checking its "
            "English Wikipedia article instead. Report the founding year and the exact source URL "
            "it was read from."
        ),
    },
]


def _make_connector():
    from shared.connector_config import ConnectorConfig

    cfg = ConnectorConfig()
    mock_backend = AsyncMock(spec=LLMBackend)
    mock_backend.normalize_payload.side_effect = lambda p, *_, **__: p
    mock_backend.simplify_payload.side_effect = lambda p: dict(p)
    with patch("agent.app.connector_llm.create_llm_backend", return_value=mock_backend):
        from agent.app.connector_llm import ConnectorLLM

        return ConnectorLLM(cfg)


class _CapturingIO:
    """Captures ``build_llm_payload``'s messages without calling any LLM. Used only to extract
    the real system prompt via the real ``LlmExpansionPolicy._build_messages`` code path."""

    def __init__(self, connector):
        self._connector = connector
        self.payload_kwargs: dict = {}

    def set_telemetry(self, telemetry):
        return None

    def build_llm_payload(self, **kwargs):
        self.payload_kwargs = kwargs
        return self._connector.build_payload(**kwargs)

    async def query_llm_with_fallback(self, payload, **kwargs):
        raise AssertionError("should not be called — this IO only captures the prompt")


def build_messages(mandate: str) -> Dict[str, str]:
    """The REAL expansion system+user prompt for a root node with this mandate, flag ON, via
    the actual ``LlmExpansionPolicy._build_messages`` code path (not a hand reconstruction)."""
    connector = _make_connector()
    io = _CapturingIO(connector)
    policy = LlmExpansionPolicy(io=io, settings={"expansion_alternative_branch_enabled": True})
    graph = IdeaDag(root_title="Research", root_details={"mandate": mandate})
    node = graph.get_node(graph.root_id())
    messages = policy._build_messages(graph, node)
    system_msgs = [m["content"] for m in messages if m.get("role") == "system"]
    user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
    assert system_msgs, "expected a system message"
    assert user_msgs, "expected a user message"
    return {"system": system_msgs[0], "user": user_msgs[0]}


def call_ollama(base_url: str, model: str, system: str, user: str, timeout: int = 120) -> str:
    resp = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"num_predict": 900, "temperature": 0.2},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return str(data.get("message", {}).get("content", ""))


def count_hits(text: str) -> Dict[str, int]:
    lower = text.lower()
    return {
        "alternative_of": lower.count("alternative_of"),
        "race_group": lower.count("race_group"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:0.5b")
    parser.add_argument("--base-url", default="http://localhost:11435")
    args = parser.parse_args()

    try:
        tags = requests.get(f"{args.base_url.rstrip('/')}/api/tags", timeout=5)
        tags.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"No local ollama instance reachable at {args.base_url}: {exc}")
        return 1

    print(f"ollama reachable at {args.base_url}, using model={args.model}\n")

    totals = {"old": {"alternative_of": 0, "race_group": 0, "any": 0},
              "new": {"alternative_of": 0, "race_group": 0, "any": 0}}

    for stim in STIMULI:
        built = build_messages(stim["mandate"])
        new_system, user = built["system"], built["user"]
        assert NEW_ADDENDUM in new_system, "new addendum not present in built prompt"
        old_system = new_system.replace(NEW_ADDENDUM, OLD_ADDENDUM)
        assert OLD_ADDENDUM in old_system

        print(f"=== stimulus: {stim['id']} ===")
        for label, system in (("old", old_system), ("new", new_system)):
            try:
                completion = call_ollama(args.base_url, args.model, system, user)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{label}] ollama call failed: {exc}")
                continue
            hits = count_hits(completion)
            any_hit = int(hits["alternative_of"] > 0 or hits["race_group"] > 0)
            totals[label]["alternative_of"] += hits["alternative_of"]
            totals[label]["race_group"] += hits["race_group"]
            totals[label]["any"] += any_hit
            print(f"  [{label}] alternative_of={hits['alternative_of']} race_group={hits['race_group']} "
                  f"(completion {len(completion)} chars)")
        print()

    n = len(STIMULI)
    print("=== summary ===")
    for label in ("old", "new"):
        t = totals[label]
        print(f"{label} addendum: {t['any']}/{n} stimuli emitted at least one of "
              f"alternative_of/race_group; total alternative_of={t['alternative_of']}, "
              f"total race_group={t['race_group']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
