import pytest
import asyncio
import logging
from app.agent import Agent

logging.basicConfig(level=logging.INFO)

@pytest.mark.asyncio
async def test_agent_find_panda_diet():
    """
    Make the agent perform a simple task, but also cite a source.
    Utilizes searching, visiting, thinking.
    """
    mandate = "Find out what pandas eat, but give a source."
    async with Agent(mandate=mandate, max_ticks=10) as agent:
        output = await agent.run()
        result = output['deliverables']
        logging.info(f"Agent deliverables: {result}")
        logging.info(f"Agent history: {output['history']}")

        string = "".join(result)
        assert "panda" in string
        assert "http" in string
        assert "eat" in string
