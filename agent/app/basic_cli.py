import asyncio
from app.agent import Agent
import logging

logging.basicConfig(level=logging.INFO)


async def main():
    print("Agent Interactive Mode")
    print("=" * 64)
    print("Enter mandates for the agent to execute.")
    print("Type 'EXIT' to quit.\n")

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
            async with Agent(mandate=mandate, max_ticks=20) as agent:
                output = await agent.run()
                logging.info(f"Agent deliverables: {output['deliverables']}")
                logging.info(f"Agent history: {output['history']}")
        except Exception as e:
            logging.error(f"Error during agent execution: {e}")

        print("\n" + "=" * 64 + "\n")
        print("\n" * 16)


if __name__ == "__main__":
    asyncio.run(main())