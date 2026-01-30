"""
Startup message utilities for services.
"""
import logging


def log_startup_message(logger: logging.Logger, service_name: str, version: str = "0.1.0") -> None:
    """
    Log formatted startup message.
    :param logger: Logger instance
    :param service_name: Service name
    :param version: Service version
    """
    logger.info("=" * 32)
    logger.info(f"**EUGLENA {service_name.upper()} STARTING**")
    logger.info("=" * 32)
    logger.info(f"Version: {version}")
    logger.info(f"Starting {service_name.title()} Service...")


def log_shutdown_message(logger: logging.Logger, service_name: str) -> None:
    """
    Log formatted shutdown message.
    :param logger: Logger instance
    :param service_name: Service name
    """
    logger.info("=" * 32)
    logger.info(f"**EUGLENA {service_name.upper()} SHUTTING DOWN**")
    logger.info("=" * 32)
