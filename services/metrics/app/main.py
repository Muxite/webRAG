import asyncio
import logging
import os
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from aiohttp import web

from shared.connector_config import ConnectorConfig
from shared.connector_rabbitmq import ConnectorRabbitMQ
from shared.health import HealthMonitor
from shared.pretty_log import setup_service_logger, log_connection_status
from shared.queue_metrics import QUEUE_DEPTH_METRIC


_rabbitmq_connector = None

def get_health_handler():
    """
    Get health check handler with properly initialized logger.
    
    :returns: Health check handler function.
    """
    logger = setup_service_logger("Metrics", logging.INFO)
    
    async def health_handler(request):
        """
        Health check endpoint handler.
        
        :param request: HTTP request object.
        :returns: JSON response with health status.
        """
        monitor = HealthMonitor(service="metrics", version="0.1.0", logger=logger)
        monitor.set_component("process", True)
        monitor.set_component("rabbitmq", _rabbitmq_connector.is_ready() if _rabbitmq_connector else False)
        monitor.log_status()
        
        from aiohttp import web
        return web.json_response(monitor.payload())
    
    return health_handler


async def _init_cloudwatch_client(logger: logging.Logger, namespace: str) -> Optional[object]:
    """
    Initialize CloudWatch client if boto3 and credentials are available.

    :param logger: Logger instance for diagnostics.
    :param namespace: CloudWatch namespace for metrics.
    :returns: CloudWatch client instance or None when unavailable.
    """

    try:
        logger.info("Initializing CloudWatch client...")
        session = boto3.Session()
        credentials = session.get_credentials()
        if credentials:
            client = boto3.client("cloudwatch")
            log_connection_status(logger, "CloudWatch", "CONNECTED", {"namespace": namespace})
            return client
        else:
            logger.warning(
                "CloudWatch metrics enabled but no AWS credentials found. "
                "Ensure task role has cloudwatch:PutMetricData permission."
            )
            return None
    except Exception as exc:
        logger.error(f"Failed to initialize CloudWatch client: {exc}", exc_info=True)
        return None


async def _publish_queue_depth_loop() -> None:
    """
    Periodically check RabbitMQ queue depth and publish to CloudWatch.
    
    :returns: None
    """
    global _rabbitmq_connector
    
    logger = setup_service_logger("MetricsService", logging.INFO)
    cfg = ConnectorConfig()

    interval_s = int(os.environ.get("QUEUE_DEPTH_METRICS_INTERVAL", "5"))
    cloudwatch_enabled = os.environ.get("PUBLISH_QUEUE_DEPTH_METRICS", "").lower() in ("1", "true", "yes")
    namespace = os.environ.get("CLOUDWATCH_NAMESPACE", QUEUE_DEPTH_METRIC.namespace)
    queue_name = os.environ.get("QUEUE_NAME", cfg.input_queue)

    logger.info(f"Starting metrics: queue={queue_name}, interval={interval_s}s, cloudwatch={cloudwatch_enabled}")
    
    rabbitmq = ConnectorRabbitMQ(cfg)
    _rabbitmq_connector = rabbitmq

    try:
        await rabbitmq.connect()
        logger.info(f"Connected to RabbitMQ: queue={queue_name}")
    except Exception as exc:
        logger.error(f"Failed to connect to RabbitMQ: {exc}", exc_info=True)
        return

    cloudwatch = None
    if cloudwatch_enabled:
        cloudwatch = await _init_cloudwatch_client(logger, namespace)
        if cloudwatch is None:
            logger.warning("CloudWatch unavailable, metrics will only be logged")

    try:
        while True:
            try:
                if not rabbitmq.is_ready():
                    logger.warning("RabbitMQ not ready, reconnecting...")
                    await rabbitmq.connect()
                
                depth = await rabbitmq.get_queue_depth(queue_name)
                
                if depth is not None:
                    logger.info(f"Queue depth: {queue_name}={depth}")
                    
                    if cloudwatch_enabled and cloudwatch is not None:
                        try:
                            await asyncio.to_thread(
                                cloudwatch.put_metric_data,
                                Namespace=namespace,
                                MetricData=[{
                                    "MetricName": QUEUE_DEPTH_METRIC.metric_name,
                                    "Dimensions": QUEUE_DEPTH_METRIC.dimensions(queue_name),
                                    "Unit": "Count",
                                    "Value": float(depth),
                                }],
                            )
                            logger.debug(f"Published to CloudWatch: {queue_name}={depth}")
                        except ClientError as exc:
                            logger.warning(f"CloudWatch publish failed: {exc}")
                        except Exception as exc:
                            logger.warning(f"CloudWatch error: {exc}")
                else:
                    logger.warning(f"Queue depth unavailable: {queue_name}")
                    
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(f"Metrics loop error: {exc}", exc_info=True)
                
            await asyncio.sleep(interval_s)
    finally:
        try:
            await rabbitmq.disconnect()
        except Exception:
            pass


async def main() -> None:
    """
    Entry point for metrics service that reports RabbitMQ queue depth.

    :returns: None
    """

    logger = setup_service_logger("Metrics", logging.INFO)
    logger.info("Starting Metrics Service...")

    app = web.Application()
    app.router.add_get("/health", get_health_handler())
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8082)
    await site.start()
    logger.info("Metrics health check server started on port 8082")

    metrics_task = None
    try:
        metrics_task = asyncio.create_task(_publish_queue_depth_loop())
        await metrics_task
    except asyncio.CancelledError:
        logger.info("Metrics service cancelled")
    except Exception as exc:
        logger.error(f"Metrics service error: {exc}", exc_info=True)
    finally:
        if metrics_task and not metrics_task.done():
            metrics_task.cancel()
            try:
                await metrics_task
            except asyncio.CancelledError:
                pass
        await runner.cleanup()
        logger.info("Metrics service stopped")


if __name__ == "__main__":
    asyncio.run(main())
