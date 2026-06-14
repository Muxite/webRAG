"""One-off: run the plain+JSON preflight gate against the full benchmark roster."""
import asyncio
import sys

from agent.app.idea_test_runner import preflight_check_llm
from agent.app.connector_llm import ConnectorLLM
from shared.connector_config import ConnectorConfig
from agent.app.testing.config import BENCHMARK_ROSTER


async def main() -> int:
    roster = (
        BENCHMARK_ROSTER["reference"]
        + BENCHMARK_ROSTER["reference_cached"]
        + BENCHMARK_ROSTER["cheap"]
        + BENCHMARK_ROSTER["experiment"]
    )
    config = ConnectorConfig()
    results = {}
    async with ConnectorLLM(config) as llm:
        for model in roster:
            ok = await preflight_check_llm(llm, model)
            results[model] = ok
            print(f"  {'PASS' if ok else 'DROP'}  {model}")
    survivors = [m for m, ok in results.items() if ok]
    print(f"\nSurvivors ({len(survivors)}/{len(roster)}): {', '.join(survivors)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
