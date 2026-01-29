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


async def main():
    print("Agent Interactive Mode")
    print("=" * 64)
    print("Enter mandates for the agent to execute.")
    print("Type 'EXIT' to quit.\n")

    config = ConnectorConfig()
    connector_llm = ConnectorLLM(config)
    connector_search = ConnectorSearch(config)
    connector_http = ConnectorHttp(config)
    connector_chroma = ConnectorChroma(config)
    
    await connector_search.init_search_api()
    await connector_chroma.init_chroma()

    while True:
        mandate = input("Enter mandate: ").strip()

        if mandate.upper() == "EXIT":
            print("Exiting agent interactive mode. Goodbye!")
            break

        if not mandate:
            print("Please enter a valid mandate.\n")
            continue

        print(f"\nExecuting mandate: {mandate}\n")
        try:
            async with Agent(
                mandate=mandate,
                max_ticks=80,
                connector_llm=connector_llm,
                connector_search=connector_search,
                connector_http=connector_http,
                connector_chroma=connector_chroma,
            ) as agent:
                output = await agent.run()
                logging.info(pretty_log_print(output))
        except Exception as e:
            logging.error(f"Error during agent execution: {e}")

        print("\n" + "=" * 64 + "\n")
        print("\n" * 16)


if __name__ == "__main__":
    asyncio.run(main())