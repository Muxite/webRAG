import asyncio
import logging
import os
from aiohttp import web
from shared.connector_config import ConnectorConfig
from shared.pretty_log import setup_service_logger, log_connection_status
from shared.startup_message import log_startup_message, log_shutdown_message
from shared.health import HealthMonitor
from agent.app.interface_agent import InterfaceAgent


async def health_handler(request):
    """
    Health check endpoint that verifies agent dependencies.
    Always returns HTTP 200 if process is running (process-level health check).
    Component status is informational but doesn't fail the health check.
    
    :param request: aiohttp request object.
    :returns: Health status with component breakdown.
    """
    logger = setup_service_logger("Agent", logging.INFO)
    monitor = HealthMonitor(service="agent", version="0.1.0", logger=logger)
    monitor.set_component("process", True)
    
    worker = request.app.get("worker")
    if worker:
        monitor.set_component("rabbitmq", worker.rabbitmq.is_ready() if hasattr(worker, 'rabbitmq') else False)
        monitor.set_component("redis", worker.storage.connector.redis_ready if hasattr(worker, 'storage') else False)
        monitor.set_component("worker_ready", worker.worker_ready if hasattr(worker, 'worker_ready') else False)
    else:
        monitor.set_component("rabbitmq", False)
        monitor.set_component("redis", False)
        monitor.set_component("worker_ready", False)
    
    payload = monitor.payload()
    monitor.log_status()
    return web.json_response(payload)


async def _run() -> None:
    logger = setup_service_logger("Agent", logging.INFO)
    
    log_startup_message(logger, "AGENT", "0.1.0")
    app = web.Application()
    app.router.add_get('/health', health_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8081)
    await site.start()
    logger.info("Agent health check server started on port 8081")
    
    config = ConnectorConfig()
    worker = InterfaceAgent(config)
    app["worker"] = worker
    
    try:
        logger.info("Initializing agent worker...")
        await worker.start()
        log_connection_status(logger, "AgentWorker", "CONNECTED")
        logger.info("Agent worker started successfully")
    except Exception as e:
        logger.error(f"Failed to start agent worker: {e}", exc_info=True)
        logger.info("Agent will continue running and health endpoint will remain available")
    
    shutdown_timeout = float(os.environ.get("AGENT_SHUTDOWN_TIMEOUT_SECONDS", "30.0"))
    try:
        while True:
            await asyncio.sleep(3600)
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Agent interrupted")
    finally:
        log_shutdown_message(logger, "AGENT")
        try:
            await asyncio.wait_for(worker.stop(), timeout=shutdown_timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Worker shutdown timed out after {shutdown_timeout}s")
        except Exception as e:
            logger.error(f"Error during worker shutdown: {e}", exc_info=True)
        try:
            await asyncio.wait_for(runner.cleanup(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Runner cleanup timed out")
        except Exception as e:
            logger.error(f"Error during runner cleanup: {e}", exc_info=True)
        logger.info("Agent Service stopped")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()