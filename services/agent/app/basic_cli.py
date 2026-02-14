import asyncio
from agent.app.agent import Agent
from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_chroma import ConnectorChroma
from shared.connector_config import ConnectorConfig
import logging
from shared.pretty_log import pretty_log_print

logging.basicConfig(level=logging.INFO)

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


def choose_model(current_model: str) -> str:
    """
    Prompt the user to select a model from the supported list.
    :param current_model: Currently selected model.
    :return: Selected model name.
    """
    print("\nAvailable models:")
    for idx, model in enumerate(MODEL_CANDIDATES, start=1):
        suffix = " (current)" if model == current_model else ""
        print(f"  [{idx}] {model}{suffix}")
    print("Enter number or model name (blank to keep current)")
    choice = input("> ").strip()
    if not choice:
        return current_model
    if choice.isdigit():
        index = int(choice) - 1
        if 0 <= index < len(MODEL_CANDIDATES):
            return MODEL_CANDIDATES[index]
        return current_model
    normalized = normalize_model_name(choice)
    if normalized in MODEL_CANDIDATES:
        return normalized
    return current_model


def parse_int(value: str, fallback: int) -> int:
    """
    Parse an integer with fallback.
    :param value: Raw value to parse.
    :param fallback: Fallback integer value.
    :return: Parsed integer.
    """
    try:
        parsed = int(value)
        return parsed if parsed > 0 else fallback
    except ValueError:
        return fallback


def print_help() -> None:
    """
    Print available CLI commands.
    :return: None
    """
    print("\nCommands:")
    print("  :help                Show this help")
    print("  :model               Choose model")
    print("  :ticks [n]           Set max ticks")
    print("  :topic [text]        Set topic prefix for mandates")
    print("  :repeat              Repeat last mandate")
    print("  :exit                Quit")
    print("")

async def main():
    print("Agent Interactive Mode")
    print("=" * 64)
    print("Enter mandates for the agent to execute.")
    print("Type :help for commands.\n")

    config = ConnectorConfig()
    connector_llm = ConnectorLLM(config)
    connector_search = ConnectorSearch(config)
    connector_http = ConnectorHttp(config)
    connector_chroma = ConnectorChroma(config)

    async with connector_search, connector_http, connector_llm:
        await connector_search.init_search_api()
        await connector_chroma.init_chroma()

        current_model = normalize_model_name(connector_llm.get_model())
        if current_model not in MODEL_CANDIDATES:
            MODEL_CANDIDATES.append(current_model)
        current_model = choose_model(current_model)
        connector_llm.set_model(current_model)
        max_ticks = 80
        topic_prefix = ""
        last_mandate = ""

        while True:
            prompt = f"[model={current_model} ticks={max_ticks}] mandate> "
            mandate = input(prompt).strip()

            if not mandate and last_mandate:
                mandate = last_mandate

            if mandate.lower() in {":exit", "exit", "quit"}:
                print("Exiting agent interactive mode. Goodbye!")
                break
            if mandate.lower() in {":help", "help"}:
                print_help()
                continue
            if mandate.lower() in {":model", "model"}:
                current_model = choose_model(current_model)
                connector_llm.set_model(current_model)
                print(f"Using model: {current_model}")
                continue
            if mandate.lower().startswith(":ticks"):
                parts = mandate.split(maxsplit=1)
                if len(parts) > 1:
                    max_ticks = parse_int(parts[1], max_ticks)
                print(f"Max ticks: {max_ticks}")
                continue
            if mandate.lower().startswith(":topic"):
                parts = mandate.split(maxsplit=1)
                topic_prefix = parts[1].strip() if len(parts) > 1 else ""
                if topic_prefix:
                    print(f"Topic prefix set: {topic_prefix}")
                else:
                    print("Topic prefix cleared")
                continue
            if mandate.lower() in {":repeat", "repeat"}:
                if not last_mandate:
                    print("No previous mandate")
                    continue
                mandate = last_mandate

            if not mandate:
                print("Please enter a valid mandate.\n")
                continue

            full_mandate = f"{topic_prefix}\n{mandate}".strip() if topic_prefix else mandate
            last_mandate = mandate
            print(f"\nExecuting mandate: {full_mandate}\n")
            try:
                async with Agent(
                    mandate=full_mandate,
                    max_ticks=max_ticks,
                    connector_llm=connector_llm,
                    connector_search=connector_search,
                    connector_http=connector_http,
                    connector_chroma=connector_chroma,
                    model_name=current_model,
                ) as agent:
                    output = await agent.run()
                    logging.info(pretty_log_print(output))
            except Exception as e:
                logging.error(f"Error during agent execution: {e}")

            print("\n" + "=" * 64 + "\n")
            print("\n" * 16)


if __name__ == "__main__":
    asyncio.run(main())