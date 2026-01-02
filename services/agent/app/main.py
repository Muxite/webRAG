import asyncio
import logging
from aiohttp import web
from shared.connector_config import ConnectorConfig
from agent.app.interface_agent import InterfaceAgent


async def health_handler(request):
    """Health check endpoint for ECS container health checks."""
    return web.json_response({"status": "healthy", "service": "agent", "version": "0.1.0"})


async def _run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("AgentMain")
    
    app = web.Application()
    app.router.add_get('/health', health_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8081)
    await site.start()
    logger.info("Agent health check server started on port 8081")
    
    config = ConnectorConfig()
    worker = InterfaceAgent(config)
    try:
        await worker.start()
        logger.info("Agent worker started successfully")
    except Exception as e:
        logger.error(f"Failed to start agent worker: {e}")
        logger.info("Agent will continue running and health endpoint will remain available")
    
    try:
        while True:
            await asyncio.sleep(3600)
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        try:
            await worker.stop()
        except Exception:
            pass
        await runner.cleanup()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()