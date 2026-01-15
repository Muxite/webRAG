import asyncio
import logging
from aiohttp import web
from shared.connector_config import ConnectorConfig
from shared.ecs_manager import EcsManager
from shared.health import HealthMonitor
from shared.pretty_log import setup_service_logger, log_connection_status
from agent.app.interface_agent import InterfaceAgent


async def health_handler(request):
    """
    Health check endpoint that verifies agent dependencies.
    
    :param request: aiohttp request object.
    :returns: Health status with component breakdown.
    """
    logger = setup_service_logger("Agent", logging.INFO)
    monitor = HealthMonitor(service="agent", version="0.1.0", logger=logger)
    monitor.set_component("process", True)
    
    worker = request.app.get("worker")
    if worker:
        monitor.set_component("rabbitmq", worker.rabbitmq.is_ready())
        monitor.set_component("redis", worker.storage.connector.redis_ready)
        monitor.set_component("worker_ready", worker.worker_ready)
    else:
        monitor.set_component("rabbitmq", False)
        monitor.set_component("redis", False)
        monitor.set_component("worker_ready", False)
    
    monitor.log_status()
    return web.json_response(monitor.payload())


async def _run() -> None:
    logger = setup_service_logger("Agent", logging.INFO)
    
    logger.info("Starting Agent Service...")
    app = web.Application()
    app.router.add_get('/health', health_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8081)
    await site.start()
    logger.info("Agent health check server started on port 8081")
    
    config = ConnectorConfig()
    ecs_manager = EcsManager()
    
    worker = InterfaceAgent(
        connector_config=config,
        ecs_manager=ecs_manager
    )
    app["worker"] = worker
    
    try:
        logger.info("Initializing agent worker...")
        await worker.start()
        log_connection_status(logger, "AgentWorker", "CONNECTED")
        logger.info("Agent worker started successfully")
    except Exception as e:
        logger.error(f"Failed to start agent worker: {e}", exc_info=True)
        logger.info("Agent will continue running and health endpoint will remain available")
    
    try:
        check_interval = 5
        while not worker.should_exit():
            await asyncio.sleep(check_interval)
        logger.info("Agent exiting due to free timeout")
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Agent interrupted")
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
