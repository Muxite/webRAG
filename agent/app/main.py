import asyncio
import logging
from shared.connector_config import ConnectorConfig
from agent.app.agent_worker import InterfaceAgent


async def _run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    config = ConnectorConfig()
    worker = InterfaceAgent(config)
    await worker.start()
    try:
        while True:
            await asyncio.sleep(3600)
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        await worker.stop()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()