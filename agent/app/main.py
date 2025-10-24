from app.agent import Agent
import asyncio

if __name__ == "__main__":
    agent = Agent()
    asyncio.run(agent.run_worker())