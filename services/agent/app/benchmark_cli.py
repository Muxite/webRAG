import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import logging

from agent.app.agent import Agent
from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_chroma import ConnectorChroma
from shared.connector_config import ConnectorConfig
from agent.app.trace_recorder import TraceRecorder
from agent.app.benchmark_writer import BenchmarkWriter


MODEL_CANDIDATES = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4o",
]

MODEL_ALIASES = {
    "gpt-5": "gpt-5",
    "gpt-5-mini": "gpt-5-mini",
    "gpt-5-nano": "gpt-5-nano",
    "gpt-4o": "gpt-4o",
}


def normalize_model_name(model_name: str) -> str:
    """
    Normalize model identifiers to canonical names.
    :param model_name: Raw model string.
    :return: Canonical model name.
    """
    candidate = model_name.strip()
    return MODEL_ALIASES.get(candidate, candidate)


def load_models_from_env() -> List[str]:
    """
    Load benchmark models from environment if present.
    :return: List of model names.
    """
    raw = os.environ.get("BENCHMARK_MODELS", "").strip()
    if not raw:
        return MODEL_CANDIDATES.copy()
    parts = [normalize_model_name(p) for p in raw.split(",")]
    return [p for p in parts if p]


def prompt_benchmark_choice() -> List[Path]:
    """
    Prompt user to select benchmark protocols.
    :return: List of protocol file paths.
    """
    base_dir = Path(__file__).resolve().parent / "benchmark_protocols"
    protocol_1 = base_dir / "benchmark_protocol_1_downfall.md"
    protocol_2 = base_dir / "benchmark_protocol_2_tensor_core_evolution.md"
    protocol_3 = base_dir / "benchmark_protocol_3_regresshion.md"
    options = {
        "1": protocol_1,
        "2": protocol_2,
        "3": protocol_3,
        "both": [protocol_1, protocol_2],
        "all": [protocol_1, protocol_2, protocol_3],
    }

    print("\nSelect benchmark:")
    print("  [1] Downfall (CVE-2022-40982)")
    print("  [2] NVIDIA Tensor Core Evolution")
    print("  [3] OpenSSH regreSSHion (CVE-2024-6387)")
    print("  [b] Both (1 + 2)")
    print("  [a] All (1 + 2 + 3)")
    choice = input("> ").strip().lower()
    if choice in {"a", "all"}:
        return options["all"]
    if choice in {"b", "both"}:
        return options["both"]
    if choice in options:
        return [options[choice]]
    return options["both"]


def load_protocol_text(path: Path) -> str:
    """
    Load protocol text from file.
    :param path: Path to protocol file.
    :return: Protocol text.
    """
    return path.read_text(encoding="utf-8").strip()


def build_benchmark_mandate(topic: str) -> str:
    """
    Build a benchmark mandate that exercises search, browsing, and vector memory.
    :param topic: Research topic.
    :return: Mandate string.
    """
    return (
        "You are running a research benchmark to evaluate model quality.\n"
        f"Topic: {topic}\n"
        "Requirements:\n"
        "1) Perform at least 6 distinct web searches with varied keywords.\n"
        "2) Visit at least 10 unique websites from those searches.\n"
        "3) Follow and visit at least 3 links found within visited websites.\n"
        "4) Extract key facts and store them in the vector database with detailed cache_update entries.\n"
        "5) Use cache_retrieve to pull previously stored facts and integrate them into later reasoning.\n"
        "6) Deliver a final report that compares at least 8 sources and highlights conflicts.\n"
        "7) Include a brief methods section describing how sources were selected.\n"
    )


def result_path(run_id: str) -> Path:
    """
    Build a file path for storing benchmark results.
    :param run_id: Run identifier.
    :return: Path to output file.
    """
    base_dir = Path(__file__).resolve().parent / "benchmark_results"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{run_id}.json"


async def run_benchmark(
    model_name: str,
    mandate: str,
    max_ticks: int,
    connector_llm: ConnectorLLM,
    connector_search: ConnectorSearch,
    connector_http: ConnectorHttp,
    connector_chroma: ConnectorChroma,
    tracer: TraceRecorder,
) -> Dict[str, Any]:
    """
    Run a single benchmark job for the given model.
    :param model_name: Model identifier.
    :param mandate: Benchmark mandate.
    :param max_ticks: Max ticks for the agent run.
    :return: Result payload.
    """
    connector_llm.set_model(model_name)
    agent = Agent(
        mandate=mandate,
        max_ticks=max_ticks,
        connector_llm=connector_llm,
        connector_search=connector_search,
        connector_http=connector_http,
        connector_chroma=connector_chroma,
        model_name=model_name,
        tracer=tracer,
    )
    started = time.time()
    output = await agent.run()
    ended = time.time()
    return {
        "model": model_name,
        "duration_seconds": round(ended - started, 2),
        "mandate": mandate,
        "output": output,
        "metrics": output.get("metrics") if isinstance(output, dict) else None,
    }


async def probe_model(connector_llm: ConnectorLLM, model_name: str) -> dict:
    """
    Probe a model with small requests to infer supported parameters.
    :param connector_llm: LLM connector.
    :param model_name: Model identifier.
    :return: Profile dict.
    """
    messages = [
        {"role": "system", "content": "You are a terse assistant."},
        {"role": "user", "content": "Reply with the word OK."},
    ]
    candidates = [
        {"max_completion_tokens": 16},
        {"temperature": 1, "max_completion_tokens": 16},
        {"temperature": 1},
        {},
    ]
    for payload in candidates:
        try:
            response = await connector_llm.client.chat.completions.create(
                model=model_name,
                messages=messages,
                **payload,
            )
            if response and getattr(response, "choices", None):
                profile = {}
                if "temperature" in payload:
                    profile["temperature"] = payload.get("temperature")
                else:
                    profile["temperature"] = None
                if "max_completion_tokens" in payload:
                    profile["use_max_completion_tokens"] = True
                return profile
        except Exception as exc:
            print(f"Preflight error for {model_name} with {payload}: {exc}")
            continue
    return {"temperature": None, "use_max_completion_tokens": True}


async def main() -> None:
    """
    Run benchmark tasks for a list of models.
    """
    max_ticks = int(os.environ.get("BENCHMARK_MAX_TICKS", "100"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    models = load_models_from_env()
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    all_results: List[Dict[str, Any]] = []

    config = ConnectorConfig()
    connector_llm = ConnectorLLM(config)
    connector_search = ConnectorSearch(config)
    connector_http = ConnectorHttp(config)
    connector_chroma = ConnectorChroma(config)

    async with connector_search, connector_http, connector_llm:
        await connector_search.init_search_api()
        await connector_chroma.init_chroma()

        model_profiles: Dict[str, dict] = {}
        for model_name in models:
            normalized = normalize_model_name(model_name)
            profile = await probe_model(connector_llm, normalized)
            connector_llm.set_model_profile(normalized, profile)
            model_profiles[normalized] = profile

        protocol_paths = prompt_benchmark_choice()
        summary_path = result_path(run_id).with_suffix(".jsonl")
        writer = BenchmarkWriter(summary_path)

        try:
            for protocol_path in protocol_paths:
                mandate = load_protocol_text(protocol_path)
                print(f"\nRunning protocol: {protocol_path.name}")
                for model_name in models:
                    normalized = normalize_model_name(model_name)
                    print(f"  Model: {normalized}")
                    run_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    trace_path = Path(__file__).resolve().parent / "benchmark_results" / f"{run_id}-{protocol_path.stem}-{normalized}-{run_stamp}.jsonl"
                    tracer = TraceRecorder(trace_path)
                    try:
                        result = await run_benchmark(
                            model_name=normalized,
                            mandate=mandate,
                            max_ticks=max_ticks,
                            connector_llm=connector_llm,
                            connector_search=connector_search,
                            connector_http=connector_http,
                            connector_chroma=connector_chroma,
                            tracer=tracer,
                        )
                        success = result.get("output", {}).get("success") if isinstance(result.get("output"), dict) else None
                        print(f"  Done: {normalized} (success={success})")
                        entry = {
                            "run_id": run_id,
                            "protocol": protocol_path.name,
                            "model": normalized,
                            "result": result,
                            "trace_file": str(trace_path),
                        }
                        writer.append(entry)
                        all_results.append(entry)
                    except Exception as exc:
                        tracer.record(
                            "run_error",
                            {"model": normalized, "protocol": protocol_path.name, "error": str(exc)},
                        )
                        writer.append(
                            {
                                "run_id": run_id,
                                "protocol": protocol_path.name,
                                "model": normalized,
                                "error": str(exc),
                                "trace_file": str(trace_path),
                            }
                        )
                        raise
                    finally:
                        tracer.close()
        finally:
            writer.close()

        out_path = result_path(run_id)
        payload = {
            "run_id": run_id,
            "models": models,
            "protocols": [p.name for p in protocol_paths],
            "model_profiles": model_profiles,
            "results": all_results,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
