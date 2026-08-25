"""
Idea Test Runner - Parallel execution with multi-model support.

Environment Variables:
- IDEA_TEST_MODE: "default" or "benchmark" (benchmark defaults: 3 models, top 8 tests, 3 runs, concurrency 3)
- IDEA_TEST_TOP_N: Number of top-priority tests to run (0 = all)
- IDEA_TEST_IDS: Comma-separated list of specific test IDs to run (e.g., "019,012,014,025") - overrides IDEA_TEST_TOP_N
- IDEA_TEST_RUNS: Repeats per test/model pair
- IDEA_TEST_CONCURRENCY: Max parallel task executions
- IDEA_TEST_MODELS: Comma-separated execution models (default: MODEL_NAME or gpt-5-mini)
- IDEA_TEST_EXECUTION_VARIANTS: Execution styles to run ("graph", "sequential", or comma-separated list)
- IDEA_TEST_LOG_LEVEL: Logging level (default: INFO)
- IDEA_TEST_BROWSER_FALLBACK: "1" (default) gives every execution variant a headless-Chrome
  fallback for bot-blocked visits (401/403/429/503, or a transport-level failure); "0"/"false"/
  "off"/"no" disables it uniformly across all arms (F18: previously silently None everywhere).

Legacy aliases still supported:
- IDEA_TEST_PRIORITY -> IDEA_TEST_TOP_N
- IDEA_TEST_REPEATS -> IDEA_TEST_RUNS
- IDEA_TEST_MAX_PARALLEL -> IDEA_TEST_CONCURRENCY
- IDEA_TEST_BENCHMARK_MODE -> IDEA_TEST_MODE=benchmark
- IDEA_TEST_BENCHMARK_THIRD_COUNT -> IDEA_TEST_TOP_N

Validation always uses gpt-5-mini regardless of execution model.
"""

import asyncio
import hashlib
import json
import os
import time
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Mapping

import logging

from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch, create_search_backend
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_chroma import ConnectorChroma
from agent.app.connector_browser import ConnectorBrowser
from shared.connector_config import ConnectorConfig
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.testing.test_module import IdeaTestModule
from agent.app.testing.runner import (
    run_complete_test,
    discover_test_modules,
    BASELINE_VARIANTS,
    OFFTHESHELF_VARIANTS,
)
from agent.app.testing.config import (
    MODEL_CANDIDATES,
    normalize_model_name,
    load_models_from_env,
    VALIDATION_MODEL,
    extract_test_id,
    filter_test_files_by_priority,
)
from agent.app.testing.utils import slim_telemetry_raw, summarize_observability
from agent.app.testing.tooling import profile_for_variant
from agent.app.idea_graph_analyzer import add_graph_visualization
from agent.app.testing.report import TestReportGenerator, Verbosity
from agent.app.trace_recorder import sanitize_model_component


def _browser_fallback_enabled() -> bool:
    """Whether execution variants get a headless-Chrome fallback connector.

    Default ON. Set IDEA_TEST_BROWSER_FALLBACK=0 to disable.
    """
    raw = os.environ.get("IDEA_TEST_BROWSER_FALLBACK", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


# Execution variants that structurally never touch ChromaDB, derived from
# testing/runner.py's own dispatch tuples (the single source of truth for which
# execution function a variant reaches) rather than a fresh hand-maintained list here:
#   - BASELINE_VARIANTS ("parametric", "naive_rag", "minimal") -> run_baseline_execution,
#     whose connector_chroma param is documented "unused; kept for signature parity"
#     (testing/execution.py).
#   - OFFTHESHELF_VARIANTS ("langgraph_react") -> run_offtheshelf_execution /
#     langgraph_solver.py, which holds a connector_chroma reference but never calls
#     store_chroma/retrieve_chroma on it.
# Every other registered variant (graph, sequential_react, naive_discretion,
# graph_compiled, graph_compiled_code, ...) DOES reach real store_chroma/retrieve_chroma
# calls (directly or via MemoryManager), so it stays in the "needs chroma" bucket.
# This is deliberately a NO-CHROMA allowlist, not a needs-chroma list: any new variant
# added to testing/runner.py (a new *_VARIANTS tuple + dispatch branch in
# run_complete_test) is "needs chroma" by default unless someone explicitly adds it
# here, so an unrecognized/future variant never silently loses its memory.
_NO_CHROMA_VARIANTS = frozenset(BASELINE_VARIANTS) | frozenset(OFFTHESHELF_VARIANTS)


def _execution_variants_need_chroma(execution_variants: List[str]) -> bool:
    """Whether ANY requested execution variant needs ChromaDB init/warmup.

    A mixed matrix (some Chroma arms, some not) must still initialize Chroma, so this
    is an ``any()`` gate, not ``all()``.
    """
    return any(variant not in _NO_CHROMA_VARIANTS for variant in execution_variants)


def _make_connector_set(config: ConnectorConfig) -> Dict[str, Any]:
    """Build one independent connector set for a concurrent test-run slot.

    Each slot gets its own ConnectorLLM to avoid race conditions when running
    different models in parallel.
    """
    return {
        "llm": ConnectorLLM(config),
        "search": create_search_backend(config),
        "http": ConnectorHttp(config),
        "chroma": ConnectorChroma(config),
        "browser": ConnectorBrowser(config),
    }


def _build_connector_pool(config: ConnectorConfig, pool_size: int) -> List[Dict[str, Any]]:
    """
    Build a pool of independent connector sets sized to the resolved concurrency.
    A pool of size 1 degenerates to exactly one shared set (today's behavior).
    :param config: Shared connector configuration.
    :param pool_size: Number of concurrent slots (resolved IDEA_TEST_CONCURRENCY).
    :return: List of connector-set dicts, length == max(1, pool_size).
    """
    return [_make_connector_set(config) for _ in range(max(1, int(pool_size)))]


def _preflight_call_timeout_seconds() -> float:
    """Per-payload-candidate timeout for the plain pre-flight probe (below).

    Default 20 preserves existing behavior for cloud API models. A local (Ollama) model can need
    much longer on any given attempt it is NOT already resident for: 20s covers a warm response but
    not a cold model-load (found live, 2026-08-07: barrage20 smoke run: qwen2.5:1.5b's own repeat
    preflight failed silently, empty-string asyncio.TimeoutError, after alternating with a different
    local model evicted it from Ollama's single-loaded-model slot, `OLLAMA_MAX_LOADED_MODELS=1`).
    IDEA_TEST_PREFLIGHT_TIMEOUT_SECONDS lets a caller (e.g. the ladder driver, for local-provider
    cells only) raise this without touching the cloud-API default.
    """
    try:
        return float(os.environ.get("IDEA_TEST_PREFLIGHT_TIMEOUT_SECONDS", "20"))
    except (TypeError, ValueError):
        return 20.0


async def preflight_check_llm(connector_llm: ConnectorLLM, model_name: str) -> bool:
    """
    Pre-flight check to verify LLM calls work for a model.
    :param connector_llm: LLM connector.
    :param model_name: Model name to test.
    :return: True if model works, False otherwise.
    """
    messages = [
        {"role": "system", "content": "You are a terse assistant."},
        {"role": "user", "content": "Reply with exactly OK"},
    ]
    payload_candidates = [
        {"max_completion_tokens": 256, "temperature": 1},
        {"max_tokens": 256, "temperature": 1},
        {"max_completion_tokens": 512},
        {"max_tokens": 512},
        {},
    ]
    original_model = connector_llm.get_model()
    last_error = "unknown"
    connector_llm.set_model(model_name)
    preflight_timeout = _preflight_call_timeout_seconds()
    try:
        working_payload = None
        for payload in payload_candidates:
            try:
                response = await asyncio.wait_for(
                    connector_llm.client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        **payload,
                    ),
                    timeout=preflight_timeout,
                )
                choices = getattr(response, "choices", None) or []
                if not choices:
                    continue
                choice0 = choices[0]
                finish_reason = getattr(choice0, "finish_reason", None)
                content = getattr(getattr(choice0, "message", None), "content", None)
                profile = {
                    "temperature": payload.get("temperature") if "temperature" in payload else None,
                    "use_max_completion_tokens": "max_completion_tokens" in payload or "max_tokens" in payload,
                }
                connector_llm.set_model_profile(model_name, profile)
                content_len = len(content.strip()) if isinstance(content, str) else 0
                logging.info(
                    f"[PASSED] Pre-flight check for {model_name} "
                    f"(finish_reason={finish_reason}, content_len={content_len}, payload={payload})"
                )
                working_payload = payload
                break
            except Exception as exc:
                last_error = str(exc)
                continue

        if working_payload is None:
            logging.warning(f"[FAILED] Pre-flight check for {model_name}: {last_error}")
            return False

        # Structured-JSON gate: the engine relies on json_mode for expansion / evaluation /
        # merge, so a model that can't emit parseable JSON cannot drive the graph. Drop it.
        if _is_enabled(os.environ.get("IDEA_TEST_PREFLIGHT_JSON", "1")):
            if not await _preflight_json_capable(connector_llm, model_name, working_payload):
                logging.warning(
                    f"[FAILED] Structured-JSON pre-flight for {model_name}: "
                    f"model could not emit parseable json_mode output (dropped)"
                )
                return False
        return True
    except Exception as exc:
        logging.error(f"[FAILED] Pre-flight check for {model_name}: {exc}")
        return False
    finally:
        connector_llm.set_model(original_model)


async def _preflight_json_capable(connector_llm: ConnectorLLM, model_name: str, base_payload: dict) -> bool:
    """
    Verify a model can produce parseable JSON under response_format=json_object.
    :param connector_llm: LLM connector.
    :param model_name: Model name to test.
    :param base_payload: The token/temperature payload that passed the plain check.
    :return: True if the model returns a parseable JSON object.
    """
    messages = [
        {"role": "system", "content": "You output only JSON. No prose."},
        {"role": "user", "content": 'Return a JSON object exactly like {"ok": true, "n": 1}'},
    ]
    payload = dict(base_payload)
    payload["response_format"] = {"type": "json_object"}
    # Reasoning models (gpt-5*, gemini-2.5-pro) spend hidden tokens before emitting
    # content; the plain check's tiny 256-token cap can starve the JSON body and cause a
    # false drop. The engine uses 8k+ tokens in practice, so give the probe real room.
    json_budget = int(os.environ.get("IDEA_TEST_PREFLIGHT_JSON_TOKENS", "4096"))
    if "max_completion_tokens" in payload:
        payload["max_completion_tokens"] = max(int(payload["max_completion_tokens"]), json_budget)
    elif "max_tokens" in payload:
        payload["max_tokens"] = max(int(payload["max_tokens"]), json_budget)
    else:
        payload["max_completion_tokens"] = json_budget
    try:
        response = await asyncio.wait_for(
            connector_llm.client.chat.completions.create(
                model=model_name,
                messages=messages,
                **payload,
            ),
            timeout=20,
        )
        choices = getattr(response, "choices", None) or []
        if not choices:
            return False
        content = getattr(getattr(choices[0], "message", None), "content", None)
        if not isinstance(content, str) or not content.strip():
            return False
        parsed = json.loads(content)
        return isinstance(parsed, dict)
    except Exception as exc:
        logging.info(f"[json-gate] {model_name} JSON probe error: {exc}")
        return False


async def _warmup_chroma(connector_chroma) -> None:
    """Warm ChromaDB (pre-install embedding models). Safe to run concurrently with preflight;
    the underlying server is shared so a single warmup suffices. Never raises."""
    logging.info("Warming up ChromaDB to pre-install embedding models...")
    try:
        warmup_collection_name = "warmup_test"
        warmup_collection = await connector_chroma.get_or_create_collection(warmup_collection_name)
        if warmup_collection:
            await connector_chroma.add_to_chroma(
                collection=warmup_collection_name,
                ids=["warmup_1"],
                metadatas=[{"type": "warmup"}],
                documents=["ChromaDB warmup document to trigger model installation"],
            )
            await connector_chroma.query_chroma(
                collection=warmup_collection_name,
                query_texts=["warmup query"],
                n_results=1,
            )
            logging.info("ChromaDB warmup completed - embedding models ready")
        else:
            logging.warning("ChromaDB warmup collection creation failed, but continuing")
    except Exception as warmup_exc:
        logging.warning(f"ChromaDB warmup failed (non-fatal): {warmup_exc}")


async def _run_preflight_parallel(
    connector_pool: List[Dict[str, Any]], preflight_models: List[str]
) -> Dict[str, bool]:
    """Preflight every model, fanning out across the pooled connector slots.

    Each concurrent check runs on a DISTINCT pooled ``ConnectorLLM`` (its ``set_model``
    mutates instance state, so two checks may never share one). Models are processed in
    waves of ``len(connector_pool)`` so a slot is only ever used by one in-flight check.
    With a pool of size 1 (IDEA_TEST_CONCURRENCY=1) this degenerates to the original serial
    loop: byte-identical attribution-mode behavior. Returns ``{model: passed}``.
    """
    pool_size = max(1, len(connector_pool))
    results: Dict[str, bool] = {}
    for i in range(0, len(preflight_models), pool_size):
        wave = preflight_models[i:i + pool_size]
        checks = [
            preflight_check_llm(connector_pool[j]["llm"], model_name)
            for j, model_name in enumerate(wave)
        ]
        wave_results = await asyncio.gather(*checks)
        for model_name, ok in zip(wave, wave_results):
            results[model_name] = ok
            if not ok:
                logging.warning(f"Skipping {model_name} - pre-flight check failed")
    return results


TEST_PRIORITY_ORDER = [
    "026",  # Deterministic Page Facts (1/10) - Priority 1 (baseline easy)
    "002",  # Basic Fact Retrieval (1/10) - Priority 2 (baseline easy)
    "025",  # Wikipedia Link Chain Game (2/10) - Priority 3 (easy link-following)
    "019",  # Explicit Visit Requirement (3/10) - Priority 4
    "033",  # Dual-Niche Comparison (5/10) - Priority 5 (branching: 2-way)
    "034",  # Laundry-List Multi-Topic Extraction (6/10) - Priority 6 (branching: 6-way fan-out)
    "035",  # Cross-Domain Niche Synthesis (7/10) - Priority 7 (branching: 3-way diverge-converge)
    "036",  # Adversarial Compare-and-Contrast (8/10) - Priority 8 (branching: 4-way + essay)
    "037",  # Five-Topic Convergence Challenge (9/10) - Priority 9 (branching: 5-way convergence)
    "038",  # Eight-Source Fact Matrix (9/10) - Priority 10 (branching: 8-way parallel extraction)
    "039",  # Multi-Branch Claim Verification (10/10) - Priority 11 (4-branch search+visit+verdict)
    "040",  # Multi-Hop Dependent Chain (8/10) - dependent 3-hop; breaks naive_rag by construction
    "041",  # Wide-Breadth Source Matrix (9/10) - 6 named sources; breaks naive_rag's 3-visit cap
    "042",  # Faithfulness & Contradiction (9/10) - planted-false claims; exercises the verify action
    "043",  # Capstone Reconciliation & Synthesis (10/10) - multi-hop breadth + judged synthesis
    "044",  # Security Vuln Root-Cause Analysis (10/10) - deep CVE research; keystone in source not summaries
    "045",  # Micro: single-page fact extraction (2/10) - level=micro, weight=short
    "046",  # Navigation: link-following traversal (7/10) - level=navigation, weight=long
    "047",  # Graph: wiki-race shortest chain (10/10) - level=graph, weight=long
    # Cross-shape tiered ladder (tier1..tier5): the graph_compiled / compiler A/B/C suite.
    "048",  # Tier 1: single-page fact (2/10) - level=micro, weight=short
    "049",  # Tier 2: two-page combine (4/10) - level=integration, weight=short
    "050",  # Tier 3: URL-free 2-hop search chain (7/10) - level=navigation, weight=long
    "051",  # Tier 4: URL-free 3-hop dependent chain (9/10) - level=graph, weight=long
    "052",  # Tier 5: 6-way fan-out & aggregation / argmin (8/10) - level=graph, weight=long
    "053",  # Tier 5: 6-way fan-out & aggregation / argmax, page-only depths (8/10) - level=graph
    "054",  # Tier 5: mixed DAG: parallel gather + dependent final hop (8/10) - level=graph
    "055",  # Reasoning+math: two 2-hop chains + terminal arithmetic difference (9/10) - level=graph
    "056",  # Cross-source contradiction resolution: fact-check w/ verify leaf (8/10) - level=integration
    "057",  # Strict JSON structured output under research load (7/10) - level=integration, weight=long
    "058",  # Long-context needle-in-haystack: buried late-section detail (7/10) - level=navigation, weight=long
    "059",  # Reasoning+math: argmax over a COMPUTED ratio (goals/caps), ranking ≠ raw (9/10) - level=graph
    "060",  # Reasoning+math: percentage-change comparison, absolute≠percent winner (9/10) - level=graph
    "061",  # Reasoning+math: two 2-hop chains (film->director->birth year) + terminal arithmetic difference (9/10) - level=graph
    "062",  # Tier 5: page-only prominence argmax; prominence≠elevation, tallest is a decoy (8/10) - level=graph
    "063",  # Strict CSV structured output under research load (7/10) - level=integration, weight=long
    "064",  # Reasoning+math: argmax over a COMPUTED ratio (volume/area), WIDE-margin double-decoy (9/10) - level=graph
    "065",  # Tier 5: URL-free 3-hop dependent chain, leak-resistant terminus (town elevation) (9/10) - level=graph, weight=long
    "066",  # Cross-source contradiction: popular-wrong vs authoritative revised record (8/10) - level=integration
    "073",  # Tier 5: temporal range filter: COUNT of founding-years within [1940,1963] (9/10) - level=integration
    "080",  # Tier 5: odd-one-out / negation B: New Guinea river drainage; Sepik is the exception (9/10) - level=integration
    "081",  # Tier 5: two-constraint numeric AND-filter: reworked to 6 self-describing leaves (9/10) - level=graph, weight=long
    "090",  # Tier 5: count-with-condition: Icelandic tunnels with length > 4,500 m (8/10) - level=graph
    "091",  # Tier 5: page-only height argmax, wide-margin fame-decoy: tallest of six Turkish dams (8/10) - level=graph
    "092",  # Tier 5: URL-free 3-hop dependent chain B, leak-resistant terminus (Pizarro -> Trujillo elevation) (9/10) - level=graph, weight=long
    "093",  # Tier 5: CVE advisory -> C source root-cause chain B (curl SOCKS5 CVE-2023-38545) (10/10) - level=graph
    "094",  # Tier 5: two-constraint numeric AND-filter: Norwegian fjords; Trondheimsfjord long-but-shallow (9/10) - level=graph, weight=long
    "095",  # Tier 5: branch-to-eliminate then chain forward: Rivers Avon (English-Channel survivor) -> Dorset Stour catchment (10/10) - level=graph, weight=long
    "096",  # Tier 5: URL-free 3-hop dependent chain C, leak-resistant terminus (Earhart -> Atchison elevation) (9/10) - level=graph, weight=long
    "097",  # Tier 5: URL-free 3-hop dependent chain D, leak-resistant terminus (Goya -> Fuendetodos elevation) (9/10) - level=graph, weight=long
    "098",  # Tier 5: branch-to-eliminate then chain forward: Royal Observatories (Cape=southern) -> McClean/Victoria refractor aperture (9/10) - level=graph, weight=long
    "099",  # Tier 5: branch-to-eliminate then chain forward: St. Stephen's cathedrals (Passau=largest organ) -> organ pipe count (9/10) - level=graph, weight=long
    "100",  # Tier 5: branch-to-eliminate (argmax) then chain forward: F1 street circuits (Baku=longest) -> Maiden Tower height (9/10) - level=graph, weight=long
    "101",  # Tier 5: branch-to-eliminate then chain forward: Cassini features (Regio on Iapetus) -> Iapetus mean radius (9/10) - level=graph, weight=long
    "102",  # Tier 5: branch-to-eliminate then chain forward: musician airports (Armstrong=jazz trumpeter) -> longest runway (9/10) - level=graph, weight=long
    "103",  # Tier 5: branch-to-eliminate then chain forward: Olympic stadiums (Montreal=inclined tower) -> Montreal Tower height (9/10) - level=graph, weight=long
    "104",  # Tier 5: branch-eliminate then chain: giant titanosaurs -> most-complete Dreadnoughtus -> scapula length 1.74 m (10/10) - level=graph, weight=long
    "105",  # Tier 5: branch-eliminate then chain: 'Star of ...' gems -> sapphire-at-AMNH Star of India -> 563.35 ct (10/10) - level=graph, weight=long
    "106",  # Tier 5: branch-eliminate then chain: Royal Botanic Gardens -> largest-canopy Kolkata -> Great Banyan prop roots 3,772 (10/10) - level=graph, weight=long
    "107",  # Tier 5: branch-eliminate then chain: tall lighthouses -> tallest-stone Île Vierge -> 360 steps / 82.5 m (10/10) - level=graph, weight=long
    "108",  # Tier 5: branch-eliminate then chain: Ytterby elements -> highest-Z Ytterbium -> melting point 824 °C (10/10) - level=graph, weight=long
    "109",  # Tier 5: branch-eliminate then chain: record waterfalls -> European Vinnufossen -> tallest single drop 575 m (10/10) - level=graph, weight=long
    "110",  # Tier 5: branch-eliminate then chain: 'Mark 1' computers -> first-commercial Ferranti Mark 1 -> valve count (9/10) - level=graph, weight=long
    "111",  # Tier 5: branch-eliminate then chain: World Marathon Majors -> record-ineligible Boston -> net elevation drop (9/10) - level=graph, weight=long
    "112",  # Tier 5: branch-eliminate then chain: Jupiter probes -> first-to-Jupiter Pioneer 10 -> antenna diameter (9/10) - level=graph, weight=long
    "113",  # Tier 5: branch-eliminate then chain: largest castles -> largest-by-area Malbork -> enclosed land area (9/10) - level=graph, weight=long
    "114",  # Tier 5: branch-eliminate then chain: oldest metros -> oldest-continental Budapest Line 1 -> length/stations (9/10) - level=graph, weight=long
    "115",  # Tier 5: branch-eliminate then chain: largest deserts (polar trap) -> Antarctic -> Vostok record low (9/10) - level=graph, weight=long
    "116",  # Tier 5: branch-eliminate then chain: impact craters (Chicxulub decoy) -> largest-verified Vredefort -> age 2.023 Ga (10/10) - level=graph, weight=long
    "117",  # Tier 5: branch-eliminate then chain: forts 'Fort George' -> post-Culloden Highland fort -> £92,673 build budget (10/10) - level=graph, weight=long
    "118",  # Tier 5: branch-eliminate then chain: ratites (ostrich decoy) -> dangerous cassowary -> inner-toe claw 12 cm (10/10) - level=graph, weight=long
    "119",  # Tier 5: branch-eliminate then chain: great bells (Tsar Bell decoy) -> ringing Mingun Bell -> 55,555 viss (10/10) - level=graph, weight=long
    "120",  # Tier 5: branch-eliminate then chain: Cleopatra's Needles (London/Paris traps) -> NY obelisk -> 112-day transit (10/10) - level=graph, weight=long
    "121",  # Tier 5: branch-eliminate then chain: record caves (Mammoth decoy) -> deepest Krubera -> mapped length 16.058 km (10/10) - level=graph, weight=long
    # Compound "stacked-axis" tasks (F34, 2026-08-06): harder IN KIND than the single-axis 10/10s:
    # each stacks two or three DISTINCT axis types inside the 6-9 visit budget. NOT part of the
    # active 59 (a separate, later live-$ decision); registered here so they are runnable/orderable.
    "146",  # Compound: 4 independent 2-hop chains + cross-branch argmax: hydro project -> reservoir -> surface area (Smallwood 6,527 km²) (10/10) - level=graph, weight=long
    "147",  # Compound: two-constraint AND-filter -> survivor -> disambiguated chain terminus: Alpine lakes -> Lake Constance -> High Rhine 165 km (10/10) - level=graph, weight=long
    "148",  # Compound: categorical survivor -> conflicting-source reconciliation -> constrained subset-sum: Kara Sea river (Ob) -> 3,700 not 5,410 -> 10,398 km (10/10) - level=graph, weight=long
    "149",  # Compound B: 4 independent 2-hop chains + cross-branch argmax: observatory -> largest telescope -> primary mirror (VLT 8.2 m) (10/10) - level=graph, weight=long
    # Race-and-merge pair (2026-08-20): ONE value, k REDUNDANT independent routes — any single route
    # suffices, so the merge is a first-past-the-post pick with corroboration rather than a
    # combination. Shaped to exercise the DAG v2 race mechanism; mid-tier by design, and NOT part of
    # the active 59.
    "150",  # Race-and-merge A: 3 redundant routes to one value: bridge main span (subject page / ranked list / no.wikipedia) (6/10) - level=graph, weight=medium
    "151",  # Race-and-merge B: 3 redundant routes to one value: medal award year (laureate page / medal page / off-Wikipedia reference) (7/10) - level=graph, weight=medium
    # Breadth fan-out re-balance (2026-08-22): the suite had exactly ONE breadth-shaped task (052),
    # which made the graph engine's parallel fan-out + merge machinery structurally unmeasurable.
    # 152 is a second, genuinely-independent 7-arm fan-out (no cross-entity dependency at all).
    "152",  # Breadth B: 7 independent one-hop page reads + argmax: mountain -> first-ascent year (latest = Vinson Massif, 1966) (8/10) - level=graph, weight=long
    "156",  # Breadth F: 7 independent one-hop page reads + count-with-condition: dams taller than 220 m (8/10) - level=graph, weight=long
    "153",  # Breadth C: 5 independent one-hop page reads + argmin: ship canal -> completion year (earliest = Erie Canal, 1825) (8/10) - level=graph, weight=long
    "154",  # Breadth D: 2 independent one-hop page reads + comparison: dam -> structural height (taller = Grande Dixence, 285 m vs Hoover 221.4 m) (7/10) - level=graph, weight=medium
    "155",  # Breadth E: 2 independent one-hop page reads + comparison: aircraft -> wingspan (wider = Hughes H-4 Hercules, 97.51 m vs An-225 88.4 m) (6/10) - level=graph, weight=medium
    "157",  # Breadth G: 7 independent one-hop page reads + count-with-condition: bridges with a main span > 1,200 m (8/10) - level=graph, weight=long
    # Mechanism suite (DAG v3 "Ledger", §8.3): 159 is the genuine NARROW-SEQUENTIAL shape — the
    # deliberate pair to the suite's wide-fan-out task. Width-1 chain: nothing is parallelisable,
    # so a breadth-oriented engine can only add cost (priced by the un-gated path_efficiency check).
    "159",  # Narrow-sequential: 4-hop dependent chain, leak-resistant terminus (lighthouse engineer -> novelist grandson -> burial mountain elevation) (9/10) - level=graph, weight=long
    # Mechanism suite (2026-08-25, DAG v3 ledger plan Sec 8.3): purpose-built tasks that make the
    # ledger/deficit-injector null result interpretable — each isolates ONE failure mechanism the
    # core24 suite never exercises.
    "302",  # Mechanism: DUPLICATED/SYNDICATED URLs — three domains republishing one Wikipedia article must count as ONE independent source; the independent authority (nps.gov) prints a different figure (9/10) - level=integration, weight=long
    "303",  # Mechanism: ENTITY COLLISION — near-duplicate names on one page: Tay rail bridge length vs Tay Road Bridge (9/10) - level=navigation, weight=medium
    "304",  # Mechanism: SOURCE CONFLICT requiring VERIFY — stale-vs-current page-vs-page temporal conflict; a supplied confident page must be checked against an independent source (8/10) - level=integration, weight=long
    "014",  # Deep Link Exploration (5/10) - Priority 12
    "020",  # GitHub Repository Analysis (4/10) - Priority 11
    "009",  # Deep Research Synthesis (9/10) - Priority 12
    "012",  # Wikipedia Link Collection (3/10) - Priority 13
    "001",  # Conflicting Information (3/10) - Priority 14
    "021",  # News Article Extraction (3/10) - Priority 15
    "004",  # Technical Specification (4/10) - Priority 16
    "013",  # Wikipedia Exploration (4/10) - Priority 17
    "022",  # Technical Documentation (5/10) - Priority 18
    "005",  # Social Media Analysis (5/10) - Priority 19
    "006",  # Obscure Historical Event (6/10) - Priority 20
    "007",  # Multi-Domain Synthesis (7/10) - Priority 21
    "008",  # Complex Data Analysis (8/10) - Priority 22
    "010",  # Extreme Synthesis (10/10) - Priority 23
    "011",  # Advanced Reasoning (9/10) - Priority 24
    "015",  # Multi-Page Synthesis (6/10) - Priority 25
    "016",  # Topic Connection Exploration (6/10) - Priority 26
    "017",  # Recursive Link Analysis (7/10) - Priority 27
    "018",  # Cross-Domain Exploration (7/10) - Priority 28
    "023",  # Sequential Data Gathering (6/10) - Priority 29
    "024",  # Research Document Analysis (7/10) - Priority 30
]


def _settings_fingerprint(settings: Dict[str, Any]) -> str:
    """Short, stable hash of an effective settings dict, for the result filename.

    Two runs under the same ``run_id``/model/variant/repeat but a different
    ``IDEA_TEST_ARM`` or ``IDEA_TEST_*`` override combination previously collided on
    an identical filename (the arm/override choice wasn't in the name at all), so the
    second run silently overwrote the first's JSON. Hashing the actual settings used
    disambiguates any override combination, however it was produced, without needing
    to enumerate every override flag here.
    :param settings: The effective settings dict for this run (post arm/env overrides).
    :return: 8 lowercase hex characters.
    """
    return hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:8]


def _is_enabled(value: str) -> bool:
    """
    Parse common truthy env values.
    :param value: Raw environment value.
    :return: True if enabled.
    """
    return value.strip().lower() in ("1", "true", "yes", "on")


# Named IDEA_TEST_ARM profiles: a curated flag set applied before the individual
# IDEA_TEST_* overrides below, so an explicit per-flag env var still wins (lets a
# researcher start from a named arm and tweak one axis without a hand-written
# settings file). Values are jsonkey -> literal value (mirrors idea_dag_settings.json
# types); "baseline" documents the shipped JSON defaults for every opt-in flag this
# module knows how to override; EXCEPT `final_require_grounding`, which every arm turns
# on: it is a validity gate, not a lever. A run that opened zero pages and answers from the
# model's weights is a hallucination whatever arm produced it, so refusing to present it as
# a researched result must not be something one arm gets and another does not.
_GOT_ARM_PROFILES: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "got_reexpand_enabled": False,
        "got_reexpand_max_iterations": 1,
        "got_step_confidence_judge_enabled": False,
        "got_step_confidence_reexpand_enabled": False,
        "got_step_confidence_reexpand_threshold": 0.5,
        "got_contract_reexpand_enabled": False,
        "got_reexpand_corrective_context_enabled": False,
        "got_backtrack_enabled": False,
        "connector_retry_on_failure_enabled": False,
        "tool_failure_recovery_enabled": False,
        "native_vote_k_enabled": False,
        "native_vote_k": 1,
        "expansion_expect_contract_enabled": False,
        "native_reasoning_effort_discipline_enabled": False,
        "price_tier_param_tiering_enabled": False,
        # pinned explicitly (2026-08-14): the JSON global default for this flag was flipped
        # to true (live-proven echo-fix, see idea_dag_settings.json), but baseline must stay
        # byte-identical to "adaptive OFF"; without this pin baseline would silently inherit
        # the new global default via the profile's partial dict .update().
        "expansion_input_output_framing_enabled": False,
        "final_require_grounding": True,   # validity gate, on in every arm
    },
    "good_adaptive": {
        "got_reexpand_enabled": True,
        "got_reexpand_max_iterations": 2,
        "got_step_confidence_judge_enabled": True,
        "got_step_confidence_reexpand_enabled": True,
        # F33: contract satisfaction, not the anti-calibrated judge score, decides when a
        # leaf re-expands. The judge stays on as decorrelated instrumentation and still
        # decides where the (free, deterministic) contract check has no verdict.
        "got_contract_reexpand_enabled": True,
        "got_reexpand_corrective_context_enabled": True,
        "tool_failure_recovery_enabled": True,
        "final_require_grounding": True,   # validity gate, on in every arm
    },
    # E1's ablation arm (ASSUMPTION_AUDIT.md PART 5): `good_adaptive` with the memory-based
    # duplicate-candidate filter removed, so a paired run against `good_adaptive` isolates dedup
    # and nothing else. Written as a profile rather than an IDEA_TEST_GOT_DEDUP env override
    # because the ladder driver selects arms, not per-arm environments; the two dicts must stay
    # in sync, which is what `idea_test_runner_got_flags_test.py` pins.
    "good_adaptive_nodedup": {
        "got_reexpand_enabled": True,
        "got_reexpand_max_iterations": 2,
        "got_step_confidence_judge_enabled": True,
        "got_step_confidence_reexpand_enabled": True,
        "got_contract_reexpand_enabled": True,
        "got_reexpand_corrective_context_enabled": True,
        "tool_failure_recovery_enabled": True,
        "final_require_grounding": True,   # validity gate, on in every arm
        "got_dedup_enabled": False,        # the one axis under test
    },
    # E3's arm (ASSUMPTION_AUDIT.md PART 5): `good_adaptive` with the memory-retrieval similarity
    # floor set to the value the offline histogram calibrated (0.40, the knee of the distribution),
    # so a paired run against `good_adaptive` isolates the floor and nothing else. Same
    # profile-not-env-override reasoning as `good_adaptive_nodedup` above.
    "good_adaptive_memfloor": {
        "got_reexpand_enabled": True,
        "got_reexpand_max_iterations": 2,
        "got_step_confidence_judge_enabled": True,
        "got_step_confidence_reexpand_enabled": True,
        "got_contract_reexpand_enabled": True,
        "got_reexpand_corrective_context_enabled": True,
        "tool_failure_recovery_enabled": True,
        "final_require_grounding": True,   # validity gate, on in every arm
        "memory_retrieval_similarity_floor": 0.40,   # the one axis under test (shipped: 0.0)
    },
    # T1-4's arm (ASSUMPTION_AUDIT.md PART 2): `good_adaptive` with the sibling visit-URL claim
    # map turned on, so a paired run against `good_adaptive` isolates the fan-out dedup and
    # nothing else. Same profile-not-env-override reasoning as the two arms above.
    "good_adaptive_urldedup": {
        "got_reexpand_enabled": True,
        "got_reexpand_max_iterations": 2,
        "got_step_confidence_judge_enabled": True,
        "got_step_confidence_reexpand_enabled": True,
        "got_contract_reexpand_enabled": True,
        "got_reexpand_corrective_context_enabled": True,
        "tool_failure_recovery_enabled": True,
        "final_require_grounding": True,   # validity gate, on in every arm
        "visit_sibling_url_dedup": True,   # the one axis under test (shipped: False)
    },
    # The strong-agent-trace bundle (project_strong_agent_trace_guidance): the three mechanisms
    # that cleared capture-replay + isolated live-probe + full end-to-end validation as a set
    # without double-penalizing the same completion (they share one enforcement path through the
    # merge consistency guard). `candidate_roster` is deliberately excluded -- it has a known,
    # accepted-as-safe residual gap that was never re-validated alongside these three, and mixing
    # a fourth axis into this arm would make a positive/negative result impossible to attribute.
    "good_adaptive_tracemech": {
        "got_reexpand_enabled": True,
        "got_reexpand_max_iterations": 2,
        "got_step_confidence_judge_enabled": True,
        "got_step_confidence_reexpand_enabled": True,
        "got_contract_reexpand_enabled": True,
        "got_reexpand_corrective_context_enabled": True,
        "tool_failure_recovery_enabled": True,
        "final_require_grounding": True,   # validity gate, on in every arm
        "merge_require_numeric_provenance_enabled": True,
        "merge_race_value_agreement_enabled": True,
        "merge_chain_closure_enabled": True,
    },
    # The "safety net" bundle: two mechanisms that each decline a premature/contradictory
    # termination rather than change what evidence is gathered -- D4's retry-after-skip
    # (merge.py) and D6's early-exit-respects-grounding (idea_engine.py). Bundled together
    # because both are pure "don't stop too early" gates with no shared code path and no
    # plausible interaction, unlike the trace-mechanism bundle above.
    "good_adaptive_safetynet": {
        "got_reexpand_enabled": True,
        "got_reexpand_max_iterations": 2,
        "got_step_confidence_judge_enabled": True,
        "got_step_confidence_reexpand_enabled": True,
        "got_contract_reexpand_enabled": True,
        "got_reexpand_corrective_context_enabled": True,
        "tool_failure_recovery_enabled": True,
        "final_require_grounding": True,   # validity gate, on in every arm
        "merge_retry_after_skip_enabled": True,
        "early_exit_respects_grounding_enabled": True,
    },
    # T1-5's arm (ASSUMPTION_AUDIT.md PART 1): `good_adaptive` with the dynamic beam's spread
    # computation reading the judge's pre-cap `raw_score` instead of the capped `score`, so a
    # paired run against `good_adaptive` isolates the beam-width signal fix and nothing else.
    "good_adaptive_beamraw": {
        "got_reexpand_enabled": True,
        "got_reexpand_max_iterations": 2,
        "got_step_confidence_judge_enabled": True,
        "got_step_confidence_reexpand_enabled": True,
        "got_contract_reexpand_enabled": True,
        "got_reexpand_corrective_context_enabled": True,
        "tool_failure_recovery_enabled": True,
        "final_require_grounding": True,   # validity gate, on in every arm
        "got_beam_spread_uses_raw_score_enabled": True,   # the one axis under test (shipped: False)
    },
    # T1-6's arm (ASSUMPTION_AUDIT.md PART 1): `good_adaptive` with backtrack turned ON (it is
    # OFF in `good_adaptive` and every other arm above) AND its dead-end threshold rescaled to
    # the observed path depth instead of the unreachable absolute constant of 5. Deliberately
    # bundles both flags rather than isolating the relative-threshold fix alone: backtrack
    # itself has never been live-validated as True at all, so a run with only the relative
    # threshold flipped and `got_backtrack_enabled` left off would measure nothing (the whole
    # mechanism stays inert either way). `backtrack_only` (bare, unrelated to this session)
    # already exists as the "does backtrack help at all with its unreachable absolute
    # threshold" control if anyone wants to separate the two questions later.
    "good_adaptive_backtrackrel": {
        "got_reexpand_enabled": True,
        "got_reexpand_max_iterations": 2,
        "got_step_confidence_judge_enabled": True,
        "got_step_confidence_reexpand_enabled": True,
        "got_contract_reexpand_enabled": True,
        "got_reexpand_corrective_context_enabled": True,
        "tool_failure_recovery_enabled": True,
        "final_require_grounding": True,   # validity gate, on in every arm
        "got_backtrack_enabled": True,
        "got_backtrack_dead_end_relative_enabled": True,
    },
    # The DAG v2 shape-spectrum arm: `good_adaptive` plus the three new axes that let a
    # structured run span parallel breadth, sequential chains, AND the middle ground of
    # "several candidate approaches, some of which fail" — a playbook note in the expansion
    # prompt, a pre-authored A->B fallback, and a concurrent race resolved to one winner at
    # the merge point.
    #
    # `got_contract_veto_requires_datum_enabled` is a genuine new REQUIREMENT of this arm,
    # not a copy: `good_adaptive` leaves it off, and the fallback's unverified-datum trigger
    # has no signal without it. `auto_parallel_siblings` and `enable_recursive_merge` already
    # default True globally, so they need no override here.
    "good_adaptive_playbook": {
        "got_reexpand_enabled": True,
        "got_reexpand_max_iterations": 2,
        "got_step_confidence_judge_enabled": True,
        "got_step_confidence_reexpand_enabled": True,
        "got_contract_reexpand_enabled": True,
        "got_reexpand_corrective_context_enabled": True,
        "tool_failure_recovery_enabled": True,
        "final_require_grounding": True,   # validity gate, on in every arm
        # axis 1: generic strategic advice, retrieved from the leak-gated library
        "strategy_library_enabled": True,
        "strategy_library_native_expansion_enabled": True,
        # axes 2+3: the branching schema, plus both consumers of what it authors
        "expansion_alternative_branch_enabled": True,
        "got_alternative_branch_promote_on_fail_enabled": True,
        "got_alternative_branch_promote_on_unverified_enabled": True,
        "got_contract_veto_requires_datum_enabled": True,
        "merge_race_winner_selection_enabled": True,
    },
    # Three single-axis ablations of the arm above (same cheap pattern as
    # `good_adaptive_nodedup`): each is `good_adaptive` plus exactly ONE of its three new
    # axes. Without them a negative combined result is unattributable to any one mechanism.
    "good_adaptive_playbook_notes_only": {
        "got_reexpand_enabled": True,
        "got_reexpand_max_iterations": 2,
        "got_step_confidence_judge_enabled": True,
        "got_step_confidence_reexpand_enabled": True,
        "got_contract_reexpand_enabled": True,
        "got_reexpand_corrective_context_enabled": True,
        "tool_failure_recovery_enabled": True,
        "final_require_grounding": True,   # validity gate, on in every arm
        "strategy_library_enabled": True,
        "strategy_library_native_expansion_enabled": True,
    },
    # The alt-branch axis carries `contract_veto_requires_datum` with it: it is that
    # trigger's signal source, not an independent lever, so splitting them would leave this
    # ablation testing half a mechanism.
    "good_adaptive_playbook_altbranch_only": {
        "got_reexpand_enabled": True,
        "got_reexpand_max_iterations": 2,
        "got_step_confidence_judge_enabled": True,
        "got_step_confidence_reexpand_enabled": True,
        "got_contract_reexpand_enabled": True,
        "got_reexpand_corrective_context_enabled": True,
        "tool_failure_recovery_enabled": True,
        "final_require_grounding": True,   # validity gate, on in every arm
        "expansion_alternative_branch_enabled": True,
        "got_alternative_branch_promote_on_fail_enabled": True,
        "got_alternative_branch_promote_on_unverified_enabled": True,
        "got_contract_veto_requires_datum_enabled": True,
    },
    # Race needs the schema flag too (nothing authors a `race_group` without it), but NOT
    # either promotion flag: an authored fallback simply retires unused here.
    "good_adaptive_playbook_race_only": {
        "got_reexpand_enabled": True,
        "got_reexpand_max_iterations": 2,
        "got_step_confidence_judge_enabled": True,
        "got_step_confidence_reexpand_enabled": True,
        "got_contract_reexpand_enabled": True,
        "got_reexpand_corrective_context_enabled": True,
        "tool_failure_recovery_enabled": True,
        "final_require_grounding": True,   # validity gate, on in every arm
        "expansion_alternative_branch_enabled": True,
        "merge_race_winner_selection_enabled": True,
    },
    "reexpand_only": {
        "got_reexpand_enabled": True,
        "got_reexpand_max_iterations": 2,
    },
    "confidence_only": {
        "got_step_confidence_judge_enabled": True,
        "got_step_confidence_reexpand_enabled": True,
    },
    "backtrack_only": {
        "got_backtrack_enabled": True,
    },
    "kvote_only": {
        "native_vote_k_enabled": True,
        "native_vote_k": 3,
    },
    "reason_first": {
        "merge_goal_evaluation_first_enabled": True,
        "verify_reason_first_enabled": True,
        "got_reexpand_followup_reason_first_enabled": True,
    },
    "full": {
        "got_reexpand_enabled": True,
        "got_reexpand_max_iterations": 2,
        "got_step_confidence_judge_enabled": True,
        "got_step_confidence_reexpand_enabled": True,
        "got_reexpand_corrective_context_enabled": True,
        "tool_failure_recovery_enabled": True,
        "native_vote_k_enabled": True,
        "native_vote_k": 3,
        "got_backtrack_enabled": True,
        "expansion_expect_contract_enabled": True,
        "native_reasoning_effort_discipline_enabled": True,
        "price_tier_param_tiering_enabled": True,
    },
    # "max_burn": the PRODUCTIVE ~5x-burn arm. It is `good_adaptive` (the proven winner)
    # with the ONE lever that carries the accuracy gain cranked (re-expansion DEPTH) plus
    # the supporting knobs that lever needs to actually execute (more steps to run the
    # deeper observe→re-expand cycles, wider hop/beam so there is more page/link space to
    # re-expand INTO), and one extra forced-grounding pass. The burn is self-limiting: it
    # only spends on the ~50-60% of runs where the follow-up detector fires, concentrating
    # compute on the chain/re-expansion archetypes where depth pays. Deliberately EXCLUDES
    # everything the barrage measured as net-negative or inert: k-vote (re-reads the same
    # evidence, overturns a correct temp-0 anchor), backtrack (never fires), expect-contract
    # (rewrites goals the cheap model can't meet → unproductive re-expansion), reasoning-effort
    # discipline (a no-op/wrong-direction here), and the price-tier 2x multiplier (a cost bug).
    # The Mode-1 residual ("right page, wrong value", ~30-42% of visited runs) is NOT reachable by
    # any grounding knob above; re-expansion fixes the wrong PAGE, not the wrong VALUE on the right
    # page. It is attacked here by the post-synthesis reconcile chain (recompute + verify + a
    # decorrelated variation ensemble), which runs only on answer-shaped tasks and fails open. Burn
    # now spends on BOTH gaps: grounding (deeper re-expansion) and synthesis (the reconcile chain).
    "max_burn": {
        # inherit good_adaptive verbatim (the proven-winning base)
        "got_reexpand_enabled": True,
        "got_step_confidence_judge_enabled": True,
        "got_step_confidence_reexpand_enabled": True,
        "got_reexpand_corrective_context_enabled": True,
        "tool_failure_recovery_enabled": True,
        "connector_retry_on_failure_enabled": True,   # keep ON in all arms (infra fairness)
        # the productive burn: deeper re-expansion + room + coverage to execute it
        "got_reexpand_max_iterations": 4,             # 2 -> 4 (the core accuracy lever)
        "max_steps": 90,                              # 50 -> 90 (or depth-4 re-expansion starves)
        "got_candidate_coverage_enabled": True,       # auto-extends steps so late re-expansions resolve
        # wider page/link space to re-expand into: cheap hops, not paid searches
        "visit_link_query_top_k": 25,                 # 15 -> 25
        "max_links_per_visit": 30,                    # 20 -> 30
        "got_beam_max": 7,                            # 5 -> 7 (effective fan-out cap under dynamic beam)
        "max_branching": 7,                           # raise the hard cap to match beam
        # one extra forced-grounding pass (attacks Mode-2 gate-zero)
        "grounding_max_replans": 3,                   # 2 -> 3
        # confidence trigger: nudge only. The judge is anti-calibrated, do not crank. F33
        # supersedes it as the control signal: contract satisfaction decides re-expansion and
        # protects a leaf that DID deliver its datum from the judge's mood.
        "got_step_confidence_reexpand_threshold": 0.55,
        "got_contract_reexpand_enabled": True,
        "final_require_grounding": True,   # validity gate, on in every arm
        # the synthesis-gap fix: post-synthesis reconcile chain (answer-shaped tasks only). The
        # variation ensemble supersedes k-vote, so k-vote stays OFF (they overlap; k-vote lost).
        "final_recompute_enabled": True,
        "final_verify_enabled": True,
        "final_variations_enabled": True,
        "final_variations_k": 3,
        # explicitly OFF: net-negative / inert / bugged (see comment above)
        "native_vote_k_enabled": False,
        "native_vote_k": 1,
        "got_backtrack_enabled": False,
        "expansion_expect_contract_enabled": False,
        "native_reasoning_effort_discipline_enabled": False,
        "price_tier_param_tiering_enabled": False,
    },
    # The deficit-driven ledger scheduler pair, never live-validated (see
    # docs/handoffs -- `task_ledger.py`/`run_policy_ledger_mode` and
    # `run_policy_deficit_driven_injection` in idea_policies/config.py; the injector is
    # `_maybe_inject_ledger_deficit_followup` in idea_engine.py, a generalization of the
    # older root-only `inject_coverage_visits` to ANY completing node). `ledgerobserve`
    # isolates the passive ledger's own overhead (computed every completion, changes
    # nothing); `ledgerdeficit` adds the actual mechanism -- a targeted SEARCH+VISIT pair
    # minted in-branch wherever the ledger still reports an unresolved named entity. Both
    # are `good_adaptive` plus exactly the ledger axis under test, same single-axis-ablation
    # pattern as `good_adaptive_nodedup` above.
    "good_adaptive_ledgerobserve": {
        "got_reexpand_enabled": True,
        "got_reexpand_max_iterations": 2,
        "got_step_confidence_judge_enabled": True,
        "got_step_confidence_reexpand_enabled": True,
        "got_contract_reexpand_enabled": True,
        "got_reexpand_corrective_context_enabled": True,
        "tool_failure_recovery_enabled": True,
        "final_require_grounding": True,   # validity gate, on in every arm
        "run_policy_ledger_mode": "observe",   # the one axis under test (shipped: "off")
    },
    "good_adaptive_ledgerdeficit": {
        "got_reexpand_enabled": True,
        "got_reexpand_max_iterations": 2,
        "got_step_confidence_judge_enabled": True,
        "got_step_confidence_reexpand_enabled": True,
        "got_contract_reexpand_enabled": True,
        "got_reexpand_corrective_context_enabled": True,
        "tool_failure_recovery_enabled": True,
        "final_require_grounding": True,   # validity gate, on in every arm
        "run_policy_ledger_mode": "observe",   # required: the injector reads the ledger
        "run_policy_deficit_driven_injection": True,   # the axis under test (shipped: False)
    },
    # Phase 0 ablation `graph_constrained_actions` (docs/DAG_V3_LEDGER_MASTER_PLAN_2026-08-25.md
    # section 3): falsifies "malformed-action instability is the main failure".
    #
    # What this arm actually turns on, precisely: `run_policy_constrained_decoding_enabled`
    # (idea_policies/config.py) gates ONE thing today -- `ActionPolicy._repair_malformed_json`
    # (idea_policies/actions.py:259) attaching a JSON schema to the bounded REPAIR re-ask that
    # fires only AFTER a response failed to parse. Three sites pass a schema: link selection
    # (`_LINK_SELECTION_REPAIR_SCHEMA`), merge synthesis, and the verify verdict. It does NOT
    # constrain the primary action/job-selection call, and it is additionally gated on
    # `supports_optional_field_json_schema` (confirmed local-Ollama backends only) -- on any
    # other backend this arm is a literal no-op. So a null result here falsifies only the
    # "repair-path decoding" slice of the malformed-action hypothesis, not the whole of it.
    "good_adaptive_constrained": {
        "got_reexpand_enabled": True,
        "got_reexpand_max_iterations": 2,
        "got_step_confidence_judge_enabled": True,
        "got_step_confidence_reexpand_enabled": True,
        "got_contract_reexpand_enabled": True,
        "got_reexpand_corrective_context_enabled": True,
        "tool_failure_recovery_enabled": True,
        "final_require_grounding": True,   # validity gate, on in every arm
        "run_policy_constrained_decoding_enabled": True,   # the axis under test (shipped: False)
    },
    # Phase 0 ablation `graph_shared_context` (same plan, section 3): falsifies "branch isolation
    # is the main failure", and it is the plan's KILL GATE -- if this arm alone closes most of the
    # gap to `sequential_react`, DAG v2's context projection gets fixed and no new engine is built.
    #
    # The axis is `run_policy_sibling_evidence_digest_enabled`, which renders
    # `LlmExpansionPolicy._build_sibling_evidence_block` (idea_policies/expansion.py) into the
    # expansion system prompt: the executed actions of every NON-ancestor node (query/URL plus its
    # key outcome), which root-ward-only expansion context otherwise hides. That hiding is the
    # diagnosed cause of task 123's churn (43 visits, the same sub-goals re-issued 5-8 times).
    # It deliberately does NOT set `run_policy_sibling_context_delta`/`run_policy_ledger_mode`:
    # that pair renders the LEDGER ROSTER and would conflate context visibility with the ledger's
    # own per-completion cost.
    "good_adaptive_sharedcontext": {
        "got_reexpand_enabled": True,
        "got_reexpand_max_iterations": 2,
        "got_step_confidence_judge_enabled": True,
        "got_step_confidence_reexpand_enabled": True,
        "got_contract_reexpand_enabled": True,
        "got_reexpand_corrective_context_enabled": True,
        "tool_failure_recovery_enabled": True,
        "final_require_grounding": True,   # validity gate, on in every arm
        "run_policy_sibling_evidence_digest_enabled": True,   # the axis under test (shipped: False)
    },
}


def _apply_got_arm_profile(
    idea_settings: Dict[str, Any], environ: Optional[Mapping[str, str]] = None
) -> Optional[str]:
    """Apply the ``IDEA_TEST_ARM`` named profile (if any) onto ``idea_settings``.

    :param idea_settings: Settings dict to mutate in place.
    :param environ: Optional environment mapping (defaults to ``os.environ``).
    :return: The resolved arm name if a known profile was applied, else ``None``.
    """
    env = os.environ if environ is None else environ
    arm_name = env.get("IDEA_TEST_ARM", "").strip()
    if not arm_name:
        return None
    profile = _GOT_ARM_PROFILES.get(arm_name)
    if profile is None:
        logging.warning(
            f"Unknown IDEA_TEST_ARM={arm_name!r}; known arms: "
            f"{sorted(_GOT_ARM_PROFILES.keys())}. No profile applied."
        )
        return None
    idea_settings.update(profile)
    return arm_name


def _apply_got_experiment_overrides(
    idea_settings: Dict[str, Any], environ: Optional[Mapping[str, str]] = None
) -> Dict[str, Any]:
    """Apply the per-run ``IDEA_TEST_GOT_*``/adaptive-mechanism toggles onto a loaded
    settings dict.

    These let a research run flip a dormant adaptive mechanism on/off without editing the
    checked-in ``idea_dag_settings.json`` default (which stays OFF). Each toggle: truthy
    (1/true/yes/on) enables, explicit falsey forces off, ABSENT leaves the current value
    (JSON default, or an ``IDEA_TEST_ARM`` profile applied first) untouched. Mutates and
    returns ``idea_settings``.

    If ``IDEA_TEST_ARM`` names a known profile, its flags are applied first; the
    individual ``IDEA_TEST_*`` overrides below are then layered on top, so an explicit
    per-flag env var always wins over the profile.

    Extracted from ``main()`` so the flag parsing is unit-testable without the harness.
    """
    env = os.environ if environ is None else environ

    _apply_got_arm_profile(idea_settings, env)

    # IDEA_TEST_GOT_REEXPAND: the follow-up-detector re-expansion (got_reexpand_enabled).
    _reexpand_override = env.get("IDEA_TEST_GOT_REEXPAND", "").strip()
    if _reexpand_override:
        idea_settings["got_reexpand_enabled"] = _is_enabled(_reexpand_override)
    # IDEA_TEST_GOT_REEXPAND_MAX_ITER: bound on re-expansion iterations
    # (got_reexpand_max_iterations).
    _reexpand_max_iter = env.get("IDEA_TEST_GOT_REEXPAND_MAX_ITER", "").strip()
    if _reexpand_max_iter:
        idea_settings["got_reexpand_max_iterations"] = max(1, int(_reexpand_max_iter))
    # IDEA_TEST_GOT_CORRECTIVE_CONTEXT: whether a re-expansion attaches corrective
    # context from the failed attempt (got_reexpand_corrective_context_enabled).
    _corrective_ctx = env.get("IDEA_TEST_GOT_CORRECTIVE_CONTEXT", "").strip()
    if _corrective_ctx:
        idea_settings["got_reexpand_corrective_context_enabled"] = _is_enabled(_corrective_ctx)
    # IDEA_TEST_GOT_STEP_CONFIDENCE_JUDGE: the decorrelated per-step confidence judge
    # (got_step_confidence_judge_enabled): the E-valuator substrate instrumentation.
    _stepconf_override = env.get("IDEA_TEST_GOT_STEP_CONFIDENCE_JUDGE", "").strip()
    if _stepconf_override:
        idea_settings["got_step_confidence_judge_enabled"] = _is_enabled(_stepconf_override)
    _stepconf_every = env.get("IDEA_TEST_GOT_STEP_CONFIDENCE_SAMPLE_EVERY", "").strip()
    if _stepconf_every:
        idea_settings["got_step_confidence_judge_sample_every"] = max(1, int(_stepconf_every))
    # IDEA_TEST_GOT_CONFIDENCE_REEXPAND: the confidence->action loop
    # (got_step_confidence_reexpand_enabled): a low decorrelated step-confidence score
    # drives a bounded re-expansion of the distrusted leaf. Requires the judge to be on.
    _confreexp_override = env.get("IDEA_TEST_GOT_CONFIDENCE_REEXPAND", "").strip()
    if _confreexp_override:
        idea_settings["got_step_confidence_reexpand_enabled"] = _is_enabled(_confreexp_override)
    # IDEA_TEST_GOT_CONFIDENCE_THRESHOLD: the score below which a step triggers the
    # confidence->action loop (got_step_confidence_reexpand_threshold).
    _confthresh_override = env.get("IDEA_TEST_GOT_CONFIDENCE_THRESHOLD", "").strip()
    if _confthresh_override:
        idea_settings["got_step_confidence_reexpand_threshold"] = float(_confthresh_override)
    # IDEA_TEST_GOT_CONTRACT_VETO_REQUIRES_DATUM: F35. Whether a SATISFIED contract may
    # veto the confidence->action loop only when it verified a measurable datum
    # (got_contract_veto_requires_datum_enabled). Subject-only satisfaction proves the leaf
    # opened a page matching its own goal's words, which every unfinished chain hop does.
    _contract_veto_override = env.get("IDEA_TEST_GOT_CONTRACT_VETO_REQUIRES_DATUM", "").strip()
    if _contract_veto_override:
        idea_settings["got_contract_veto_requires_datum_enabled"] = _is_enabled(
            _contract_veto_override
        )
    # IDEA_TEST_GOT_BACKTRACK: whether the graph can backtrack off a dead-end branch
    # (got_backtrack_enabled).
    _backtrack_override = env.get("IDEA_TEST_GOT_BACKTRACK", "").strip()
    if _backtrack_override:
        idea_settings["got_backtrack_enabled"] = _is_enabled(_backtrack_override)
    # IDEA_TEST_NATIVE_EARLY_EXIT: A6 calibrated high-confidence early exit
    # (native_confidence_early_exit_enabled): the symmetric counterpart of the
    # confidence->re-expansion loop, stopping an easy run instead of extending a shaky one.
    # Requires the step-confidence judge to be on (no judged steps -> no rule input), and it
    # only ever fires if the shipped calibration artifact certifies a rule. No threshold
    # override exists on purpose: confidence_early_exit_calibration.json owns the derived
    # thresholds, and a hand-set engine-level bar is exactly what the calibration replaces.
    _earlyexit_override = env.get("IDEA_TEST_NATIVE_EARLY_EXIT", "").strip()
    if _earlyexit_override:
        idea_settings["native_confidence_early_exit_enabled"] = _is_enabled(_earlyexit_override)
    _earlyexit_margin = env.get("IDEA_TEST_NATIVE_EARLY_EXIT_MARGIN", "").strip()
    if _earlyexit_margin:
        idea_settings["native_confidence_early_exit_margin"] = float(_earlyexit_margin)
    # IDEA_TEST_CONNECTOR_RETRY: whether a failed connector call is retried
    # (connector_retry_on_failure_enabled).
    _connretry_override = env.get("IDEA_TEST_CONNECTOR_RETRY", "").strip()
    if _connretry_override:
        idea_settings["connector_retry_on_failure_enabled"] = _is_enabled(_connretry_override)
    # IDEA_TEST_TOOL_FAILURE_RECOVERY: whether a failed tool call triggers a recovery
    # path (tool_failure_recovery_enabled).
    _toolrecov_override = env.get("IDEA_TEST_TOOL_FAILURE_RECOVERY", "").strip()
    if _toolrecov_override:
        idea_settings["tool_failure_recovery_enabled"] = _is_enabled(_toolrecov_override)
    # IDEA_TEST_NATIVE_VOTE_K: self-consistency vote count (native_vote_k); a value
    # >=2 also flips native_vote_k_enabled on (a k of 1 is a no-op vote, so it leaves
    # the enabled flag alone rather than forcing it off).
    _votek_override = env.get("IDEA_TEST_NATIVE_VOTE_K", "").strip()
    if _votek_override:
        _votek = max(1, int(_votek_override))
        idea_settings["native_vote_k"] = _votek
        if _votek >= 2:
            idea_settings["native_vote_k_enabled"] = True
    # IDEA_TEST_NATIVE_VOTE_K_TIERED: capability-tier the vote count
    # (native_vote_k_tiered_enabled) instead of a single fixed k.
    _votek_tiered_override = env.get("IDEA_TEST_NATIVE_VOTE_K_TIERED", "").strip()
    if _votek_tiered_override:
        idea_settings["native_vote_k_tiered_enabled"] = _is_enabled(_votek_tiered_override)
    # IDEA_TEST_NATIVE_VOTE_K_SIZE_BAND: refine the tiered weak-band vote count by LOCAL
    # model size (native_vote_k_size_band_enabled) -- a no-op unless the tiered flag above
    # is also on.
    _votek_sizeband_override = env.get("IDEA_TEST_NATIVE_VOTE_K_SIZE_BAND", "").strip()
    if _votek_sizeband_override:
        idea_settings["native_vote_k_size_band_enabled"] = _is_enabled(_votek_sizeband_override)
    # IDEA_TEST_EXPECT_CONTRACT: the expansion "expect contract" schema requirement
    # (expansion_expect_contract_enabled).
    _expectcontract_override = env.get("IDEA_TEST_EXPECT_CONTRACT", "").strip()
    if _expectcontract_override:
        idea_settings["expansion_expect_contract_enabled"] = _is_enabled(_expectcontract_override)
    # IDEA_TEST_MERGE_GOAL_EVAL_FIRST: reorder merge's goal-evaluation prompt/schema to
    # reasoning-before-verdict (merge_goal_evaluation_first_enabled).
    _merge_goal_eval_first_override = env.get("IDEA_TEST_MERGE_GOAL_EVAL_FIRST", "").strip()
    if _merge_goal_eval_first_override:
        idea_settings["merge_goal_evaluation_first_enabled"] = _is_enabled(_merge_goal_eval_first_override)
    # IDEA_TEST_VERIFY_REASON_FIRST: reorder verify's system prompt to
    # reasoning-before-verdict (verify_reason_first_enabled).
    _verify_reason_first_override = env.get("IDEA_TEST_VERIFY_REASON_FIRST", "").strip()
    if _verify_reason_first_override:
        idea_settings["verify_reason_first_enabled"] = _is_enabled(_verify_reason_first_override)
    # IDEA_TEST_GOT_FOLLOWUP_REASON_FIRST: reorder the re-expand followup prompt to
    # reasoning-before-verdict (got_reexpand_followup_reason_first_enabled).
    _followup_reason_first_override = env.get("IDEA_TEST_GOT_FOLLOWUP_REASON_FIRST", "").strip()
    if _followup_reason_first_override:
        idea_settings["got_reexpand_followup_reason_first_enabled"] = _is_enabled(
            _followup_reason_first_override
        )
    # IDEA_TEST_EXPANSION_IO_FRAMING: label the expansion user prompt's context blob as read-only
    # INPUT and restate the {candidates: [...]} output shape right after it
    # (expansion_input_output_framing_enabled): the prompt-hygiene fix for a weak model echoing
    # the context back instead of planning.
    _ioframing_override = env.get("IDEA_TEST_EXPANSION_IO_FRAMING", "").strip()
    if _ioframing_override:
        idea_settings["expansion_input_output_framing_enabled"] = _is_enabled(_ioframing_override)
    # IDEA_TEST_EXPANSION_ECHO_RETRY: one bounded corrective retry when an expansion reply parses
    # but is the echoed input rather than a plan (expansion_echo_retry_enabled). Independent of
    # the framing flag above so the prompt fix and the safety net can be ablated separately.
    _echoretry_override = env.get("IDEA_TEST_EXPANSION_ECHO_RETRY", "").strip()
    if _echoretry_override:
        idea_settings["expansion_echo_retry_enabled"] = _is_enabled(_echoretry_override)
    # IDEA_TEST_GOT_DEDUP: the memory-based duplicate-candidate filter (got_dedup_enabled,
    # default True. NOT one of the opt-in A1-A5 adaptive levers, it is baseline engine
    # behavior). Exists so a benchmark can isolate cross-rep memory-persistence effects: the
    # per-mandate memory namespace (agent/app/idea_engine.py's
    # ``idea_dag:{sha256(mandate)[:10]}``) is keyed on mandate TEXT only, not run_id or rep
    # number, so it is NOT cleared between R>1 reps of the same task. A candidate set that is
    # reproducible rep-to-rep (e.g. a plan-library template's deterministic fill) can score
    # >=dedup_threshold_min against its OWN prior rep's stored memory: found during the
    # 2026-07-28/29 plan-library dogfooding run. Since Cycle 18 an all-flagged batch passes
    # through whole instead of collapsing to one, so this costs recall rather than plan steps.
    # Toggle here to control for it.
    _gotdedup_override = env.get("IDEA_TEST_GOT_DEDUP", "").strip()
    if _gotdedup_override:
        idea_settings["got_dedup_enabled"] = _is_enabled(_gotdedup_override)
    # IDEA_TEST_PLAN_LIBRARY / _AUTO / _ACTION: retrieval-augmented planning: the master
    # switch (plan_library_enabled), the automatic pre-expansion short-circuit
    # (plan_library_auto_enabled) and the on-demand `plan_library_search` leaf action
    # (plan_library_action_enabled). The master switch plus at least one sub-switch must be on
    # for a template to reach the graph; no threshold override exists on purpose (retrieval.py
    # owns the calibrated decision constants, and a second engine-level gate would silently
    # disagree with them).
    _planlib_override = env.get("IDEA_TEST_PLAN_LIBRARY", "").strip()
    if _planlib_override:
        idea_settings["plan_library_enabled"] = _is_enabled(_planlib_override)
    _planlib_auto_override = env.get("IDEA_TEST_PLAN_LIBRARY_AUTO", "").strip()
    if _planlib_auto_override:
        idea_settings["plan_library_auto_enabled"] = _is_enabled(_planlib_auto_override)
    _planlib_action_override = env.get("IDEA_TEST_PLAN_LIBRARY_ACTION", "").strip()
    if _planlib_action_override:
        idea_settings["plan_library_action_enabled"] = _is_enabled(_planlib_action_override)
    # IDEA_TEST_STRATEGY_LIBRARY: the OTHER library: generalized prose advice spliced into the
    # graph_compiled authoring/aggregation prompts (strategy_library_enabled). One switch, no
    # threshold override: strategy_library/retrieval.py owns that constant, exactly as
    # plan_library/retrieval.py does. This is the knob an advice-on/advice-off A/B flips (see
    # scripts/eval_strategy_library_generalization.py).
    _stratlib_override = env.get("IDEA_TEST_STRATEGY_LIBRARY", "").strip()
    if _stratlib_override:
        idea_settings["strategy_library_enabled"] = _is_enabled(_stratlib_override)
    # IDEA_TEST_NATIVE_REASONING_DISCIPLINE: native reasoning-effort discipline
    # (native_reasoning_effort_discipline_enabled).
    _reasondisc_override = env.get("IDEA_TEST_NATIVE_REASONING_DISCIPLINE", "").strip()
    if _reasondisc_override:
        idea_settings["native_reasoning_effort_discipline_enabled"] = _is_enabled(_reasondisc_override)
    # IDEA_TEST_PRICE_TIER_TIERING: per-price-tier parameter tiering
    # (price_tier_param_tiering_enabled).
    _pricetier_override = env.get("IDEA_TEST_PRICE_TIER_TIERING", "").strip()
    if _pricetier_override:
        idea_settings["price_tier_param_tiering_enabled"] = _is_enabled(_pricetier_override)
    return idea_settings


def _unique_models(models: List[str]) -> List[str]:
    """
    Preserve order while removing duplicates/empties.
    :param models: Candidate model names.
    :return: Unique model names in original order.
    """
    unique: List[str] = []
    seen = set()
    for model_name in models:
        normalized = normalize_model_name(str(model_name or "").strip())
        if not normalized or normalized in seen:
            continue
        unique.append(normalized)
        seen.add(normalized)
    return unique


def _env_int(primary_key: str, fallback_keys: List[str], default_value: int) -> int:
    """
    Read integer from env using primary key with optional legacy fallbacks.
    :param primary_key: Preferred env var.
    :param fallback_keys: Legacy env vars in precedence order.
    :param default_value: Default when unset/invalid.
    :return: Parsed integer value.
    """
    raw = os.environ.get(primary_key, "").strip()
    if not raw:
        for key in fallback_keys:
            legacy = os.environ.get(key, "").strip()
            if legacy:
                raw = legacy
                break
    if raw:
        compact = raw.replace(",", "").replace("_", "").strip()
        signless = compact[1:] if compact.startswith(("+", "-")) else compact
        if signless.isdigit():
            return int(compact)
    return default_value


def _parse_execution_variants(raw: str) -> List[str]:
    """
    Parse execution variants from env.
    :param raw: Raw env string.
    :return: Ordered variant list.
    """
    text = (raw or "").strip().lower()
    if not text:
        return ["graph"]
    normalized = text.replace(";", ",").replace("\n", ",")
    parts = [item.strip() for item in normalized.split(",") if item.strip()]
    aliases = {
        "graph": "graph",
        "dag": "graph",
        "parallel": "graph",
        "sequential": "sequential",
        "chain": "sequential",
        "cot": "sequential",
        # No-graph baselines for the cost-recovery benchmark.
        "parametric": "parametric",
        "completion": "parametric",
        "raw": "parametric",
        "naive_rag": "naive_rag",
        "naive": "naive_rag",
        "rag": "naive_rag",
        # Tooling-ablation rungs (testing/tooling.py): minimal / partial / full.
        "minimal": "minimal",
        "search_only": "minimal",
        "searchonly": "minimal",
        "partial": "naive_rag",
        "full": "graph",
        # Strong linear comparator: ReAct loop, same toolset as the graph, no GoT.
        "sequential_react": "sequential_react",
        "react": "sequential_react",
        "linear": "sequential_react",
        # Honest floor: same tools and same dispatch, minimal prompt, no engineered structure.
        "naive_discretion": "naive_discretion",
        "discretion": "naive_discretion",
        "floor": "naive_discretion",
        # Cheap model executing an expensive-model-authored offline plan (no runtime planning).
        "graph_compiled": "graph_compiled",
        "compiled": "graph_compiled",
        "compiled_graph": "graph_compiled",
        # Same, in the CODE domain: the plan's leaves drive sandbox actions (write_file /
        # run_pytest / ...) instead of web search+visit. See testing/execution_compiled_code.py.
        "graph_compiled_code": "graph_compiled_code",
        "compiled_code": "graph_compiled_code",
        "code": "graph_compiled_code",
        # Off-the-shelf, publicly-available agent-system comparison arm (DAG v2 relaunch item 4):
        # a genuinely third-party orchestration loop (LangGraph), not this repo's own ReAct code.
        "langgraph_react": "langgraph_react",
        "langgraph": "langgraph_react",
        "offtheshelf": "langgraph_react",
    }
    out: List[str] = []
    seen = set()
    for part in parts:
        variant = aliases.get(part, "")
        if not variant or variant in seen:
            continue
        out.append(variant)
        seen.add(variant)
    return out or ["graph"]


def _variant_settings(base_settings: Dict[str, Any], variant: str) -> Dict[str, Any]:
    """
    Build settings for execution variant.
    :param base_settings: Base settings.
    :param variant: Variant key.
    :return: Variant-specific settings.
    """
    settings = dict(base_settings)
    if variant == "sequential":
        settings["allow_execute_all_children"] = False
        settings["best_first_global"] = False
        settings["sequential_prune_siblings"] = True

        settings["max_total_nodes"] = min(int(settings.get("max_total_nodes", 500)), 80)
        settings["max_observation_chars"] = min(int(settings.get("max_observation_chars", 100000)), 40000)
        settings["max_links_per_visit"] = min(int(settings.get("max_links_per_visit", 20)), 5)

        settings["got_dynamic_beam_enabled"] = False

        system_prompt = str(settings.get("expansion_system_prompt", "")).strip()
        if system_prompt:
            settings["expansion_system_prompt"] = (
                system_prompt
                + "\n\nSEQUENTIAL MODE: Propose multiple candidates. Only the highest-scoring one will be selected."
            )
    return settings


def _parse_effort_tiers(raw: str) -> List[int]:
    """
    Parse effort tiers (node budgets) for the recovery-curve sweep.

    Each tier is a max-node budget for the graph; ``0`` means "no override"
    (use the settings defaults, used for full-capability reference runs).
    :param raw: Comma/space separated integers.
    :return: Ordered list of tiers (default ``[0]``).
    """
    text = (raw or "").strip()
    if not text:
        return [0]
    out: List[int] = []
    seen = set()
    for part in text.replace(";", ",").replace(" ", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            continue
        if value < 0 or value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out or [0]


def _apply_effort_tier(settings: Dict[str, Any], tier_nodes: int) -> Dict[str, Any]:
    """
    Apply an effort tier (node budget) to a settings copy.

    Bounds spend by capping total nodes and step count; realized USD (measured
    ex-post) is the actual x-axis of the recovery curve.
    :param settings: Base settings.
    :param tier_nodes: Node budget; ``0`` leaves settings untouched.
    :return: Settings copy.
    """
    s = dict(settings)
    if tier_nodes and tier_nodes > 0:
        s["max_total_nodes"] = int(tier_nodes)
        steps_factor = float(os.environ.get("IDEA_TEST_TIER_STEPS_FACTOR", "2"))
        s["max_steps"] = max(10, int(round(tier_nodes * steps_factor)))
    return s


def _apply_lean_overlay(settings: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Cap token budgets + branching so the graph is cost-competitive on EASY tasks.

    The pilot showed the graph burning ~164k tokens / $0.0275 on a single-fact micro
    task (huge ``final_max_tokens``, over-expansion). For short-``weight`` tasks we shrink
    the budgets; hard (``long``) tasks keep full budgets so deep multi-hop work is not
    starved. Force on/off via ``IDEA_TEST_LEAN=1/0`` (default: auto by weight).
    :param settings: Settings copy (post effort-tier).
    :param metadata: Test metadata (uses ``weight``).
    :return: Settings copy, possibly capped.
    """
    weight = str(metadata.get("weight", "")).strip().lower()
    force = os.environ.get("IDEA_TEST_LEAN", "").strip().lower()
    if force in ("0", "false", "no", "off"):
        return settings
    lean = force in ("1", "true", "yes", "on") or weight == "short"
    if not lean:
        return settings
    s = dict(settings)
    caps = {
        "final_max_tokens": 4096,
        "expansion_max_tokens": 2048,
        "evaluation_max_tokens": 2048,
        "max_branching": 3,
        "max_total_nodes": 40,
        "max_steps": 12,
        # Big pages (e.g. Wikipedia, 500k+ chars raw) make VisitLeafAction's link
        # extraction + Chroma embedding the dominant cost; and it can exceed the 20s
        # visit timeout. Easy tasks need a fact, not the whole link graph, so cap it.
        "max_links_per_visit": 5,
        # The real token driver: full page content (64k+ chars) flows into the MERGE
        # call (observed merged_json ~400k chars -> ~100k tokens). Cap the observation
        # size fed to LLM stages AND the merge aggregation so easy tasks stay cheap.
        "max_observation_chars": 8000,
        "merge_max_json_chars": 24000,
    }
    for key, cap in caps.items():
        cur = s.get(key)
        s[key] = cap if not isinstance(cur, int) else min(int(cur), cap)
    return s


def _difficulty_value(metadata: Dict[str, Any]) -> int:
    """
    Parse integer difficulty from metadata.
    :param metadata: Test metadata.
    :return: Difficulty in range 1..10, default 5.
    """
    raw = str(metadata.get("difficulty_level", "")).strip()
    if "/" in raw:
        raw = raw.split("/", 1)[0].strip()
    try:
        value = int(raw)
    except Exception:
        value = 5
    return max(1, min(10, value))


def _is_visit_focused(test_module: IdeaTestModule) -> bool:
    """
    Determine whether a test strongly exercises visit behavior.
    :param test_module: Loaded test module wrapper.
    :return: True when visit behavior is explicit.
    """
    metadata = test_module.metadata or {}
    category = str(metadata.get("category", "")).lower()
    statement = str(test_module.get_task_statement() or "").lower()
    criteria = " ".join(str(item).lower() for item in (test_module.get_success_criteria() or []))
    text = f"{category} {statement} {criteria}"
    return "visit" in text or "must visit" in text or "url" in text


def select_benchmark_test_files(test_files: List[Path], target_count: int = 8) -> List[Path]:
    """
    Select a balanced benchmark subset with strong priority and difficulty spread.
    :param test_files: Available test files.
    :param target_count: Target number of tests.
    :return: Selected tests in priority order.
    """
    if not test_files:
        return []
    test_id_to_file = {extract_test_id(f): f for f in test_files}
    ordered = [test_id_to_file[test_id] for test_id in TEST_PRIORITY_ORDER if test_id in test_id_to_file]
    for test_file in test_files:
        if test_file not in ordered:
            ordered.append(test_file)
    target = max(1, min(target_count, len(ordered)))
    modules: Dict[Path, IdeaTestModule] = {}
    for test_file in ordered:
        modules[test_file] = IdeaTestModule(test_file)
    visit_candidates = [f for f in ordered if _is_visit_focused(modules[f])]
    selected: List[Path] = []
    for candidate in visit_candidates[: min(2, target)]:
        if candidate not in selected:
            selected.append(candidate)
    bucket_targets: Dict[str, int]
    if target == 8:
        bucket_targets = {"low": 3, "mid": 3, "high": 2}
    else:
        low = max(1, round(target * 0.34))
        mid = max(1, round(target * 0.33))
        high = max(1, target - low - mid)
        bucket_targets = {"low": low, "mid": mid, "high": high}
    def bucket_name(value: int) -> str:
        if value <= 4:
            return "low"
        if value <= 7:
            return "mid"
        return "high"
    bucket_counts = {"low": 0, "mid": 0, "high": 0}
    for item in selected:
        b = bucket_name(_difficulty_value(modules[item].metadata))
        bucket_counts[b] += 1
    for candidate in ordered:
        if len(selected) >= target:
            break
        if candidate in selected:
            continue
        b = bucket_name(_difficulty_value(modules[candidate].metadata))
        if bucket_counts[b] < bucket_targets[b]:
            selected.append(candidate)
            bucket_counts[b] += 1
    for candidate in ordered:
        if len(selected) >= target:
            break
        if candidate not in selected:
            selected.append(candidate)
    selected_ids = [extract_test_id(f) for f in selected]
    logging.info(f"Benchmark mode: selected {len(selected)} tests: {selected_ids}")
    logging.info(
        "Benchmark mix: "
        f"low={bucket_counts['low']}, mid={bucket_counts['mid']}, high={bucket_counts['high']}, "
        f"visit_focused={sum(1 for f in selected if _is_visit_focused(modules[f]))}"
    )
    return selected


def extract_test_id(test_file: Path) -> str:
    """
    Extract test ID from test file name.
    :param test_file: Test file path.
    :return: Test ID string (e.g., "001", "020").
    """
    stem = test_file.stem
    if stem.startswith("test_"):
        remaining = stem.replace("test_", "", 1)
        parts = remaining.split("_", 1)
        return parts[0] if parts else ""
    return ""


def filter_test_files_by_priority(test_files: List[Path], priority_count: int = 0, specific_ids: Optional[List[str]] = None) -> List[Path]:
    """
    Filter test files by priority order or specific test IDs.
    :param test_files: List of test file paths.
    :param priority_count: Number of priority tests to run (0 = all, ignored if specific_ids provided).
    :param specific_ids: Optional list of specific test IDs to run (e.g., ["019", "012"]).
    :return: Filtered list of test files in priority order.
    """
    test_id_to_file = {extract_test_id(f): f for f in test_files}
    
    if specific_ids:
        filtered = []
        for test_id in specific_ids:
            test_id = test_id.strip()
            if test_id in test_id_to_file:
                filtered.append(test_id_to_file[test_id])
            else:
                logging.warning(f"Test ID {test_id} not found, skipping")
        logging.info(f"Specific test mode: Running {len(filtered)} specified tests: {[extract_test_id(f) for f in filtered]}")
        return filtered
    
    if priority_count == 0:
        ordered = []
        for test_id in TEST_PRIORITY_ORDER:
            if test_id in test_id_to_file:
                ordered.append(test_id_to_file[test_id])
        for test_file in test_files:
            test_id = extract_test_id(test_file)
            if test_id not in TEST_PRIORITY_ORDER:
                ordered.append(test_file)
        logging.info(f"Running all {len(ordered)} tests in priority order")
        return ordered
    
    filtered = []
    for test_id in TEST_PRIORITY_ORDER[:priority_count]:
        if test_id in test_id_to_file:
            filtered.append(test_id_to_file[test_id])
    
    logging.info(f"Priority mode: Running top {len(filtered)} priority tests: {[extract_test_id(f) for f in filtered]}")
    return filtered


def _result_cost_usd(result: Dict[str, Any]) -> float:
    """Total measured USD for one cell: runtime observability cost + any offline compiler cost.

    Mirrors what ``recovery_curve._load_row`` reads (``execution.observability.cost.usd``) and
    adds the ``compiler`` block that ``graph_compiled`` records for offline plan authoring, so the
    spend ceiling reflects true cumulative spend. Missing/unpriced costs count as 0.0.
    """
    total = 0.0
    execution = result.get("execution") or {}
    try:
        total += float((execution.get("observability") or {}).get("cost", {}).get("usd") or 0.0)
    except (AttributeError, TypeError, ValueError):
        pass
    try:
        # graph_compiled records the offline plan-authoring cost under execution.compiler.cost.
        total += float((execution.get("compiler") or {}).get("cost", {}).get("usd") or 0.0)
    except (AttributeError, TypeError, ValueError):
        pass
    return total


_LOCAL_LLM_PROVIDERS = ("ollama", "local")


def _result_origin(connector_llm: ConnectorLLM) -> str:
    """"local" if this run's backend targets a self-hosted provider (Ollama/local), else "api".

    Stamped at the source (from the actual resolved provider config) rather than inferred
    downstream from missing pricing data, so reporting code can distinguish "no price
    applies" from "price data is just missing"; see model_costs.is_local_row.
    """
    provider = (getattr(getattr(connector_llm, "config", None), "llm_provider", None) or "").strip().lower()
    return "local" if provider in _LOCAL_LLM_PROVIDERS else "api"


async def run_single_test(
    test_file: Path,
    model_name: str,
    connector_llm: ConnectorLLM,
    connector_search: ConnectorSearch,
    connector_http: ConnectorHttp,
    connector_chroma: ConnectorChroma,
    idea_settings: Dict[str, Any],
    run_id: str,
    results_dir: Path,
    validation_model: str,
    execution_variant: str = "graph",
    repeat_index: int = 1,
    total_repeats: int = 1,
    effort_tier: int = 0,
    connector_browser: Optional[ConnectorBrowser] = None,
) -> Dict[str, Any]:
    """
    Run a single test for a single model.
    :param connector_browser: Optional headless-Chrome fallback connector, threaded into
        every execution variant's ``AgentIO`` uniformly (F18: None disables it uniformly
        instead of leaving it silently inconsistent across variants).
    :return: Test result dict or None if failed.
    """
    test_id = extract_test_id(test_file)
    normalized = normalize_model_name(model_name)

    try:
        test_module = IdeaTestModule(test_file)
        metadata = test_module.metadata

        variant_specific_settings = _variant_settings(idea_settings, execution_variant)
        variant_specific_settings = _apply_effort_tier(variant_specific_settings, effort_tier)
        variant_specific_settings = _apply_lean_overlay(variant_specific_settings, metadata)

        # Computed BEFORE the run so the trace path can carry the same disambiguating suffix
        # as the result JSON below. Without it, repeats and A/B conditions of one cell shared
        # a trace path -- and TraceRecorder appends, so retained traces would interleave.
        tier_tag = f"_t{effort_tier}" if effort_tier else ""
        cfg_tag = f"_cfg{_settings_fingerprint(variant_specific_settings)}"
        cell_tag = f"{tier_tag}{cfg_tag}_r{repeat_index}"

        result = await run_complete_test(
            test_module=test_module,
            model_name=normalized,
            connector_llm=connector_llm,
            connector_search=connector_search,
            connector_http=connector_http,
            connector_chroma=connector_chroma,
            connector_browser=connector_browser,
            idea_settings=variant_specific_settings,
            run_stamp=run_id,
            summarize_observability_func=summarize_observability,
            validation_model=validation_model,
            execution_variant=execution_variant,
            cell_tag=cell_tag,
        )

        result["execution_variant"] = execution_variant
        result["tooling_profile"] = profile_for_variant(execution_variant)
        result["effort_tier"] = effort_tier
        result["origin"] = _result_origin(connector_llm)

        result = add_graph_visualization(result)

        # Shared with scripts/eval_strategy_library_generalization.py and
        # scripts/unified_bench_report.py via agent.app.trace_recorder.sanitize_model_component
        # so the one "/" -> "-" (":" untouched) convention can't drift into three copies again.
        # Output must stay byte-identical to the old `normalized.replace("/", "-")` for every
        # real model id -- see agent/tests/idea_test_runner_result_filename_test.py.
        safe_model = sanitize_model_component(normalized)
        out_path = (
            results_dir
            / f"{run_id}_{test_id}_{safe_model}_{execution_variant}{tier_tag}{cfg_tag}_r{repeat_index}.json"
        )
        
        report_verbosity = _env_int("IDEA_TEST_REPORT_VERBOSITY", [], 1)
        reporter = TestReportGenerator(verbosity=report_verbosity)
        telemetry_raw = result.get("execution", {}).get("telemetry_raw")
        report = reporter.generate(result, telemetry_data=telemetry_raw)
        reporter.print_console(report)
        if report_verbosity >= Verbosity.STANDARD:
            reporter.save(report, out_path)
        
        if report_verbosity < Verbosity.FULL:
            # Slim rather than drop. Popping the whole block below FULL, combined with every
            # variant unlinking its JSONL trace on success, left a default run with NO
            # per-step record in either place -- 0 of 96 cells in the four-way baseline had
            # one. The size that motivated the pop lives in the page bodies and embeddings
            # that `slim_telemetry_raw` removes, not in the timings/decisions/events that
            # offline forensics actually reads.
            execution = result.get("execution", {})
            if "telemetry_raw" in execution:
                execution["telemetry_raw"] = slim_telemetry_raw(execution["telemetry_raw"])
        
        # Atomic write: a crash mid-write (Ctrl-C, OOM, the driver's 1800s timeout kill) must never
        # leave a truncated/partial result JSON on disk. A partial file there would still satisfy a
        # naive "does the file exist" resume check, so the cell would be silently skipped forever
        # with no score and no cost ever recorded. Write to a sibling temp file, then atomically
        # rename into place (os.replace) so the final path is always either absent or complete.
        result_text = json.dumps(result, indent=2, default=str)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp_path.write_text(result_text, encoding="utf-8")
        os.replace(tmp_path, out_path)

        # Opt-in: emit a square >=1920px PNG of the run's DAG (compiled plan or runtime graph)
        # alongside the result JSON. Best-effort: visualization must never fail a run.
        if os.environ.get("IDEA_TEST_RENDER_DAG", "").lower() in ("1", "true", "yes"):
            try:
                from agent.app.testing.dag_visualizer import render_result_file
                dag_px = _env_int("IDEA_TEST_DAG_PX", [], 1920)
                render_result_file(str(out_path), out_path=str(out_path.with_suffix(".dag.png")),
                                   side_px=dag_px)
            except Exception as exc:  # noqa: BLE001
                logging.debug(f"[{test_id}] DAG render skipped: {exc}")

        validation = result.get("validation", {})
        passed = validation.get("overall_passed", False)
        score = validation.get("overall_score", 0.0)
        duration = result.get("execution", {}).get("duration_seconds", 0)
        observability = result.get("execution", {}).get("observability", {})
        
        logging.info(
            f"[{test_id}] {normalized} [{execution_variant}] [run {repeat_index}/{total_repeats}]: "
            f"{'PASSED' if passed else 'FAILED'} (score: {score:.2f}, {duration:.1f}s)"
        )
        
        return result
        
    except Exception as exc:
        logging.error(f"[{test_id}] {normalized} failed: {exc}", exc_info=True)
        return {
            "test_metadata": {"test_id": test_id},
            "model": normalized,
            "execution_variant": execution_variant,
            "tooling_profile": profile_for_variant(execution_variant),
            "origin": _result_origin(connector_llm),
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat(),
        }


def _print_run_summary(all_results: List[Dict[str, Any]], run_id: str, priority_count: int) -> None:
    """
    Print a consolidated summary table of all test results to the console.

    :param all_results: List of all test result dicts.
    :param run_id: Run identifier.
    :param priority_count: Number of priority tests (0=all).
    """
    sep = "=" * 80
    logging.info(f"\n{sep}")
    logging.info(f"  RUN SUMMARY: {run_id}")
    logging.info(sep)
    logging.info(f"  {'Test':>6}  {'Model':>14}  {'Variant':>10}  {'Pass':>5}  {'Score':>6}  {'Visits':>6}  {'Nodes':>6}  {'Time':>7}")
    logging.info(f"  {'------':>6}  {'--------------':>14}  {'----------':>10}  {'-----':>5}  {'------':>6}  {'------':>6}  {'------':>6}  {'-------':>7}")

    total_passed = 0
    total_tests = 0
    total_score = 0.0

    for r in all_results:
        meta = r.get("test_metadata", {})
        test_id = meta.get("test_id", "???")
        model = r.get("model", "?")
        variant = r.get("execution_variant", "?")
        val = r.get("validation", {})
        passed = val.get("overall_passed", False)
        score = val.get("overall_score", 0.0)
        exec_data = r.get("execution", {})
        obs = exec_data.get("observability", {})
        visit_count = obs.get("visit", {}).get("count", 0)
        graph = exec_data.get("graph", {})
        node_count = len(graph.get("nodes", {}))
        duration = exec_data.get("duration_seconds", 0)

        icon = "  OK" if passed else "FAIL"
        logging.info(
            f"  {test_id:>6}  {model:>14}  {variant:>10}  {icon:>5}  {score:>6.2f}  {visit_count:>6}  {node_count:>6}  {duration:>6.1f}s"
        )
        total_tests += 1
        total_score += score
        if passed:
            total_passed += 1

    avg_score = total_score / total_tests if total_tests > 0 else 0.0
    logging.info(sep)
    logging.info(f"  Passed: {total_passed}/{total_tests}  Avg Score: {avg_score:.2f}")
    logging.info(f"  Priority mode: {priority_count} tests (0 = all)")

    from collections import defaultdict
    mv_scores: Dict[str, list] = defaultdict(list)
    mv_passed: Dict[str, int] = defaultdict(int)
    mv_count: Dict[str, int] = defaultdict(int)
    for r in all_results:
        model = r.get("model", "?")
        variant = r.get("execution_variant", "?")
        key = f"{model} [{variant}]"
        val = r.get("validation", {})
        score = val.get("overall_score", 0.0)
        mv_scores[key].append(score)
        mv_count[key] += 1
        if val.get("overall_passed", False):
            mv_passed[key] += 1

    if mv_count:
        logging.info("")
        logging.info("  Per Model+Variant Breakdown:")
        logging.info(f"  {'System':>30}  {'Runs':>4}  {'Pass%':>6}  {'Avg':>6}  {'Med':>6}  {'Min':>6}  {'Max':>6}")
        logging.info(f"  {'-'*30:>30}  {'----':>4}  {'-----':>6}  {'-----':>6}  {'-----':>6}  {'-----':>6}  {'-----':>6}")
        ranked = sorted(mv_count.keys(), key=lambda k: sum(mv_scores[k]) / len(mv_scores[k]), reverse=True)
        for key in ranked:
            n = mv_count[key]
            sc = mv_scores[key]
            import numpy as np
            avg = np.mean(sc)
            med = np.median(sc)
            mn = np.min(sc)
            mx = np.max(sc)
            pr = mv_passed[key] / n if n > 0 else 0
            logging.info(f"  {key:>30}  {n:>4}  {pr:>5.1%}  {avg:>6.3f}  {med:>6.3f}  {mn:>6.3f}  {mx:>6.3f}")

        graph_keys = [k for k in ranked if "[graph]" in k]
        seq_keys = [k for k in ranked if "[sequential]" in k]
        if graph_keys:
            best_g = graph_keys[0]
            g_avg = sum(mv_scores[best_g]) / len(mv_scores[best_g])
            logging.info(f"\n  >>> BEST GRAPH: {best_g} (avg {g_avg:.3f})")
        if seq_keys:
            best_s = seq_keys[0]
            s_avg = sum(mv_scores[best_s]) / len(mv_scores[best_s])
            logging.info(f"  >>> BEST SEQUENTIAL: {best_s} (avg {s_avg:.3f})")
        if graph_keys and seq_keys:
            g_avg = sum(mv_scores[graph_keys[0]]) / len(mv_scores[graph_keys[0]])
            s_avg = sum(mv_scores[seq_keys[0]]) / len(mv_scores[seq_keys[0]])
            winner = graph_keys[0] if g_avg >= s_avg else seq_keys[0]
            logging.info(f"  >>> OVERALL BEST: {winner}")

    aggregated_timings: Dict[str, Dict[str, Any]] = {}
    for r in all_results:
        obs = r.get("execution", {}).get("observability", {})
        for name, stats in obs.get("timings", {}).items():
            if name not in aggregated_timings:
                aggregated_timings[name] = {
                    "count": 0,
                    "total_duration": 0.0,
                    "min_duration": float("inf"),
                    "max_duration": 0.0,
                }
            agg = aggregated_timings[name]
            agg["count"] += stats.get("count", 0)
            agg["total_duration"] += stats.get("total_duration", 0)
            agg["min_duration"] = min(agg["min_duration"], stats.get("min_duration", float("inf")))
            agg["max_duration"] = max(agg["max_duration"], stats.get("max_duration", 0))
    for agg in aggregated_timings.values():
        if agg["min_duration"] == float("inf"):
            agg["min_duration"] = 0.0
        agg["avg_duration"] = round(agg["total_duration"] / agg["count"], 4) if agg["count"] > 0 else 0.0
        agg["total_duration"] = round(agg["total_duration"], 4)
        agg["min_duration"] = round(agg["min_duration"], 4)
        agg["max_duration"] = round(agg["max_duration"], 4)

    if aggregated_timings:
        from agent.app.testing.report import _render_timing_bar_chart
        logging.info("")
        logging.info("  Aggregate Connector Timings Across All Tests:")
        for name, stats in sorted(aggregated_timings.items()):
            logging.info(
                f"    {name:>16}: avg={stats['avg_duration']:.3f}s  "
                f"min={stats['min_duration']:.3f}s  max={stats['max_duration']:.3f}s  "
                f"count={stats['count']}  total={stats['total_duration']:.2f}s"
            )
        chart_lines = _render_timing_bar_chart(aggregated_timings, title="Avg Duration (all tests)")
        for line in chart_lines:
            logging.info(line)

    logging.info(sep)


async def main() -> None:
    """Run idea test suite with parallel execution."""
    mode_env = os.environ.get("IDEA_TEST_MODE", "").strip().lower()
    benchmark_mode = mode_env == "benchmark" or _is_enabled(os.environ.get("IDEA_TEST_BENCHMARK_MODE", ""))
    
    log_level = os.environ.get("IDEA_TEST_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    
    models = _unique_models(load_models_from_env())
    validation_model = normalize_model_name(
        os.environ.get("IDEA_TEST_VALIDATION_MODEL", "").strip() or VALIDATION_MODEL
    )
    if benchmark_mode:
        if len(models) < 3:
            models = _unique_models(MODEL_CANDIDATES[:3])
        if len(models) > 3:
            models = models[:3]
    # Pin a run_id via IDEA_TEST_RUN_ID so a multi-stage driver can tag every file from one
    # experiment with the same prefix (clean `--run-id` analysis); else timestamp it.
    run_id = os.environ.get("IDEA_TEST_RUN_ID", "").strip() or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    default_top_n = 8 if benchmark_mode else 0
    default_runs = 3 if benchmark_mode else 1
    default_concurrency = 3 if benchmark_mode else 4
    priority_count = _env_int("IDEA_TEST_TOP_N", ["IDEA_TEST_PRIORITY", "IDEA_TEST_BENCHMARK_THIRD_COUNT"], default_top_n)
    repeats = max(1, _env_int("IDEA_TEST_RUNS", ["IDEA_TEST_REPEATS"], default_runs))
    max_parallel = max(1, _env_int("IDEA_TEST_CONCURRENCY", ["IDEA_TEST_MAX_PARALLEL"], default_concurrency))
    # Spend kill-switch for unattended high-spend runs: abort the matrix once cumulative measured
    # cost (runtime + offline compiler) crosses this USD ceiling. 0 / unset = no ceiling.
    try:
        usd_ceiling = float(os.environ.get("IDEA_TEST_USD_CEILING", "0").strip() or 0.0)
    except ValueError:
        usd_ceiling = 0.0
    specific_test_ids = None
    test_ids_env = os.environ.get("IDEA_TEST_IDS", "").strip()
    if test_ids_env:
        specific_test_ids = [tid.strip() for tid in test_ids_env.split(",") if tid.strip()]
        logging.info(f"Filtering to specific test IDs: {specific_test_ids}")
    
    # Tooling-ablation axis. IDEA_TEST_TOOLING ("minimal,partial,full") is the
    # benchmark-facing name; it feeds the same variant parser and takes precedence
    # over the legacy IDEA_TEST_EXECUTION_VARIANTS when set.
    execution_variants_env = (
        os.environ.get("IDEA_TEST_TOOLING")
        or os.environ.get("IDEA_TEST_EXECUTION_VARIANTS", "graph")
    )
    execution_variants = _parse_execution_variants(execution_variants_env)

    effort_tiers = _parse_effort_tiers(os.environ.get("IDEA_TEST_EFFORT_TIERS", "0"))
    
    if benchmark_mode:
        max_parallel = min(max_parallel, 3)
    
    all_test_files = discover_test_modules()
    if benchmark_mode:
        test_files = select_benchmark_test_files(all_test_files, target_count=priority_count)
    else:
        test_files = filter_test_files_by_priority(all_test_files, priority_count=priority_count, specific_ids=specific_test_ids)
    
    if not test_files:
        logging.error("No test files found after filtering")
        return
    
    logging.info(f"Running {len(test_files)} test(s) (total available: {len(all_test_files)})")
    logging.info(f"Mode: {'benchmark' if benchmark_mode else 'default'}")
    logging.info(f"Models: {', '.join(models)}")
    logging.info(f"Validation model: {validation_model}")
    logging.info(f"Execution variants: {', '.join(execution_variants)} ({len(execution_variants)} variant(s))")
    logging.info(f"Effort tiers (node budgets, 0=default): {effort_tiers}")
    if specific_test_ids:
        logging.info(f"Test ID filter active: {specific_test_ids}")
    logging.info(f"Top N tests: {priority_count} (0 means all)")
    logging.info(f"Max parallel: {max_parallel}")
    logging.info(f"Repeats: {repeats}")
    if usd_ceiling > 0:
        logging.info(f"Spend ceiling: ${usd_ceiling:.4f} (matrix aborts when cumulative cost crosses it)")
    if benchmark_mode:
        logging.info("Benchmark mode enabled: balanced difficulty subset with explicit visit coverage")
    
    config = ConnectorConfig()
    # IDEA_DAG_SETTINGS_PATH: optional path to an alternate settings JSON (e.g. an
    # experimental prompt variant), so A/B prompt experiments never touch the
    # production idea_dag_settings.json default.
    _settings_path_override = os.environ.get("IDEA_DAG_SETTINGS_PATH", "").strip()
    idea_settings = load_idea_dag_settings(Path(_settings_path_override) if _settings_path_override else None)
    # IDEA_TEST_ARM: optional named profile (e.g. "good_adaptive", "full") applied
    # before the per-flag IDEA_TEST_* overrides below, so an explicit flag still wins.
    _resolved_arm = os.environ.get("IDEA_TEST_ARM", "").strip() or None
    # Per-run IDEA_TEST_GOT_*/adaptive-mechanism toggles for the dormant adaptive
    # mechanisms (re-expansion, step-confidence judge, confidence->action loop,
    # backtrack, self-consistency voting, etc). Each stays OFF in the checked-in JSON
    # default; an env toggle (or IDEA_TEST_ARM profile) flips it for a research run
    # without editing that file.
    _apply_got_experiment_overrides(idea_settings)
    _enabled_adaptive_flags = sorted(
        key
        for key in _GOT_ARM_PROFILES["full"].keys()
        if idea_settings.get(key) is True
    )
    logging.info(
        f"Idea DAG settings source: {_settings_path_override or '(default idea_dag_settings.json)'} "
        f"| arm={_resolved_arm or '(none)'}"
        f" | got_reexpand_enabled={idea_settings.get('got_reexpand_enabled')}"
        f" | got_step_confidence_judge_enabled={idea_settings.get('got_step_confidence_judge_enabled')}"
        f" | got_step_confidence_reexpand_enabled={idea_settings.get('got_step_confidence_reexpand_enabled')}"
        f" | enabled_adaptive_flags={_enabled_adaptive_flags}"
    )
    idea_settings["log_dag_ascii"] = False
    idea_settings["log_dag_step_interval"] = 0
    idea_settings["allowed_actions"] = ["search", "visit", "save", "think", "merge", "verify"]
    # Intra-graph action parallelism: the connectors are shared across leaves, and
    # large-page parsing is CPU-bound, so concurrent visits can starve the loop and
    # time out (breaks wide-breadth tasks). Allow serializing via env for reliable runs.
    _pal = os.environ.get("IDEA_TEST_PARALLEL_ACTION_LIMIT")
    if _pal:
        idea_settings["parallel_action_limit"] = max(1, int(_pal))
    # Optional visit/fetch/action/LLM timeout overrides for slow or contended calls.
    # IDEA_TEST_LLM_TIMEOUT lowers the per-call LLM wire timeout for faster-fail throughput
    # runs; unset -> the shipped llm_timeout_seconds default (success path unchanged).
    # IDEA_TEST_EXPANSION_TIMEOUT/IDEA_TEST_FINAL_TIMEOUT added 2026-08-07 (barrage20 smoke
    # finding): idea_engine.py's decision/expansion call site uses
    # `self._cfg.timeouts.expansion or self._cfg.timeouts.llm`: expansion (default 180) wins
    # whenever set, so raising IDEA_TEST_LLM_TIMEOUT alone never reaches this call at all. A
    # local (Ollama) model generating against a large completion budget on a single consumer GPU
    # can genuinely need more than 180s; without this override every such call hit the SAME
    # 180.1s ceiling regardless of LLM_READ_TIMEOUT, killing every local-model cell in the first
    # live smoke run with a clean rc=0 and zero result (see also final_timeout_seconds, which
    # already self-scales with prompt size and hard-caps at 600s; override it too for symmetry,
    # though it was not the one observed failing).
    for _env, _key in (
        ("IDEA_TEST_VISIT_TIMEOUT", "visit_timeout_seconds"),
        ("IDEA_TEST_FETCH_TIMEOUT", "fetch_timeout_seconds"),
        ("IDEA_TEST_ACTION_TIMEOUT", "action_timeout_seconds"),
        ("IDEA_TEST_LLM_TIMEOUT", "llm_timeout_seconds"),
        ("IDEA_TEST_EXPANSION_TIMEOUT", "expansion_timeout_seconds"),
        ("IDEA_TEST_FINAL_TIMEOUT", "final_timeout_seconds"),
    ):
        _val = os.environ.get(_env)
        if _val:
            idea_settings[_key] = float(_val)
    if benchmark_mode:
        visit_prompt_suffix = (
            " Benchmark rule: when external evidence is needed, run search then visit the best URLs. "
            "Visit pages to extract concrete details and cite visited URLs in the final answer."
        )
        os.environ["IDEA_TEST_MANDATE_SUFFIX"] = visit_prompt_suffix
    
    # One independent connector set per concurrent slot. IDEA_TEST_CONCURRENCY has been
    # historically pinned to 1 because a single shared ConnectorLLM.set_model() mutates
    # shared model state and races when two different-model runs overlap. A pool of size
    # `max_parallel` gives each concurrent task its own ConnectorLLM/Search/Http/Chroma,
    # so distinct-model runs can execute simultaneously without corrupting attribution.
    # Chroma clients are isolated too (err toward isolation); the underlying Chroma server
    # is shared and handles concurrent requests, so only ONE warmup is needed server-side.
    pool_size = max(1, int(max_parallel))
    connector_pool = _build_connector_pool(config, pool_size)
    # Preflight/warmup run on the first set; its populated model_profiles are copied to
    # the rest so every slot shares the same per-model runtime profiles.
    primary = connector_pool[0]
    connector_llm = primary["llm"]
    connector_search = primary["search"]
    connector_http = primary["http"]
    connector_chroma = primary["chroma"]

    browser_fallback_enabled = _browser_fallback_enabled()
    logging.info(
        f"Browser fallback (401/403/429/503 bot-block recovery): "
        f"{'ENABLED for every arm' if browser_fallback_enabled else 'DISABLED (IDEA_TEST_BROWSER_FALLBACK=0)'}"
    )

    # Chroma's init/warmup is the dominant per-cell setup cost (SentenceTransformer/CUDA cold
    # load, ~5s) and runs unconditionally even for arms that never touch ChromaDB (parametric,
    # naive_rag, minimal, langgraph_react). The requested execution arm(s) are already known
    # here (execution_variants, parsed above), so skip the init/warmup calls entirely when none
    # of them need Chroma. The ConnectorChroma object itself is still constructed in every pool
    # slot (_make_connector_set) unconditionally, for signature parity across the pool.
    chroma_needed = _execution_variants_need_chroma(execution_variants)
    if not chroma_needed:
        logging.info(
            "ChromaDB init/warmup SKIPPED: none of the requested execution variants "
            f"({', '.join(execution_variants)}) use ChromaDB."
        )

    async with AsyncExitStack() as stack:
        for cset in connector_pool:
            await stack.enter_async_context(cset["search"])
            await stack.enter_async_context(cset["http"])
            await stack.enter_async_context(cset["llm"])
            await cset["search"].init_search_api()
            if chroma_needed:
                await cset["chroma"].init_chroma()
            # Registered unconditionally (safe no-op if the browser never launched) so toggling
            # IDEA_TEST_BROWSER_FALLBACK never needs a teardown-logic change too.
            stack.push_async_callback(cset["browser"].close)

        execution_models_requested = _unique_models(models)
        preflight_models = list(execution_models_requested)
        if validation_model not in preflight_models:
            preflight_models.append(validation_model)
        logging.info(f"Pre-flight target models: {', '.join(preflight_models)}")

        # Overlap the ChromaDB warmup with preflight (both are fixed setup latency), and fan
        # preflight out across the pooled connector slots. At concurrency=1 the pool has one
        # slot so preflight stays serial (byte-identical); the warmup still just runs alongside.
        # Skipped entirely when chroma_needed is False (see above): no arm in this run touches
        # ChromaDB, so there is nothing to warm.
        warmup_task = asyncio.create_task(_warmup_chroma(connector_chroma)) if chroma_needed else None
        preflight_passed = await _run_preflight_parallel(connector_pool, preflight_models)
        if warmup_task is not None:
            await warmup_task

        execution_models = [model for model in execution_models_requested if preflight_passed.get(model, False)]
        if not execution_models:
            logging.error("No valid execution models after pre-flight checks. Aborting.")
            return
        if not preflight_passed.get(validation_model, False):
            logging.error(f"Validation model {validation_model} failed pre-flight check. Aborting.")
            return
        valid_models = _unique_models(execution_models + [validation_model])
        logging.info(f"Valid models: {', '.join(valid_models)}")

        # Parallel preflight populated per-model runtime profiles on WHICHEVER slot ran each
        # model, so consolidate: merge every slot's profiles, then broadcast the union to all
        # slots. Payload normalization reads self.model_profiles per-instance, so every pooled
        # ConnectorLLM must carry the full set. (At pool size 1 this is a no-op self-copy.)
        merged_profiles: Dict[str, Any] = {}
        for cset in connector_pool:
            merged_profiles.update(cset["llm"].model_profiles)
        for cset in connector_pool:
            for _mname, _profile in merged_profiles.items():
                cset["llm"].set_model_profile(_mname, _profile)

        # Checkout queue of connector sets, one per concurrent slot. A task borrows a set
        # for the duration of its run and returns it, guaranteeing no two concurrent tasks
        # ever share a ConnectorLLM (whose set_model mutates instance state).
        connector_queue: asyncio.Queue = asyncio.Queue()
        for cset in connector_pool:
            connector_queue.put_nowait(cset)

        all_results = []
        results_dir = Path(__file__).resolve().parent.parent / "idea_test_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        test_tasks: List[Tuple[Path, str, str, int, int]] = []
        for test_file in test_files:
            for model_name in execution_models:
                for execution_variant in execution_variants:
                    # Baselines + linear agents ignore effort tiers (single fixed pass); engine variants sweep.
                    tiers = [0] if execution_variant in ("parametric", "naive_rag", "minimal", "sequential_react", "naive_discretion") else effort_tiers
                    for effort_tier in tiers:
                        for repeat_index in range(1, repeats + 1):
                            test_tasks.append((test_file, model_name, execution_variant, effort_tier, repeat_index))
        
        logging.info(f"Total test tasks: {len(test_tasks)}")
        
        semaphore = asyncio.Semaphore(max_parallel)
        # Cumulative-spend kill-switch. Shared across cells; once tripped, queued cells
        # short-circuit (no further LLM spend). Meaningful under concurrency=1 (the barrage
        # default) where cells run effectively serially through the semaphore.
        spend_state = {"total_usd": 0.0, "tripped": False}

        async def run_with_semaphore(task):
            test_file, model_name, execution_variant, effort_tier, repeat_index = task
            queue_wait_start = time.perf_counter()
            async with semaphore:
                if usd_ceiling > 0 and spend_state["tripped"]:
                    logging.warning(
                        f"[SPEND CEILING] skipping {extract_test_id(test_file)} {model_name} "
                        f"{execution_variant} t{effort_tier} r{repeat_index} "
                        f"(cumulative ${spend_state['total_usd']:.4f} >= ${usd_ceiling:.4f})"
                    )
                    return None
                queue_wait_end = time.perf_counter()
                queue_wait_seconds = max(0.0, queue_wait_end - queue_wait_start)
                cset = await connector_queue.get()
                try:
                    result = await run_single_test(
                        test_file=test_file,
                        model_name=model_name,
                        connector_llm=cset["llm"],
                        connector_search=cset["search"],
                        connector_http=cset["http"],
                        connector_chroma=cset["chroma"],
                        connector_browser=cset["browser"] if browser_fallback_enabled else None,
                        idea_settings=idea_settings,
                        run_id=run_id,
                        results_dir=results_dir,
                        validation_model=validation_model,
                        execution_variant=execution_variant,
                        repeat_index=repeat_index,
                        total_repeats=repeats,
                        effort_tier=effort_tier,
                    )
                finally:
                    connector_queue.put_nowait(cset)
                if result and isinstance(result, dict):
                    execution_data = result.get("execution", {})
                    if isinstance(execution_data, dict):
                        execution_data["queue_wait_seconds"] = round(queue_wait_seconds, 2)
                    if usd_ceiling > 0:
                        spend_state["total_usd"] += _result_cost_usd(result)
                        if spend_state["total_usd"] >= usd_ceiling and not spend_state["tripped"]:
                            spend_state["tripped"] = True
                            logging.error(
                                f"[SPEND CEILING] cumulative cost ${spend_state['total_usd']:.4f} "
                                f">= ${usd_ceiling:.4f}: aborting remaining cells."
                            )
                return result
        
        logging.info("Starting parallel test execution...")
        results = await asyncio.gather(*[run_with_semaphore(task) for task in test_tasks], return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"Test task failed with exception: {result}")
                continue
            if result:
                all_results.append(result)
        
        summary_path = results_dir / f"{run_id}_summary.json"
        summary = {
            "run_id": run_id,
            "benchmark_mode": benchmark_mode,
            "models": valid_models,
            "validation_model": validation_model,
            "execution_variants": execution_variants,
            "effort_tiers": effort_tiers,
            "tests_run": len(test_files),
            "total_tests_available": len(all_test_files),
            "priority_count": priority_count,
            "repeats": repeats,
            "max_parallel": max_parallel,
            "test_ids": [extract_test_id(f) for f in test_files],
            "results": all_results,
            "timestamp": datetime.utcnow().isoformat(),
        }
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        
        _print_run_summary(all_results, run_id, priority_count)


if __name__ == "__main__":
    asyncio.run(main())
