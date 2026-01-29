import logging
import pytest
from shared.startup_message import log_startup_message, log_shutdown_message


def test_log_startup_message(caplog):
    logger = logging.getLogger("test_service")
    logger.setLevel(logging.INFO)
    
    log_startup_message(logger, "test", "1.0.0")
    
    assert "EUGLENA TEST STARTING" in caplog.text
    assert "Version: 1.0.0" in caplog.text
    assert "Starting Test Service..." in caplog.text


def test_log_startup_message_default_version(caplog):
    logger = logging.getLogger("test_service")
    logger.setLevel(logging.INFO)
    
    log_startup_message(logger, "agent")
    
    assert "EUGLENA AGENT STARTING" in caplog.text
    assert "Version: 0.1.0" in caplog.text


def test_log_shutdown_message(caplog):
    logger = logging.getLogger("test_service")
    logger.setLevel(logging.INFO)
    
    log_shutdown_message(logger, "gateway")
    
    assert "EUGLENA GATEWAY SHUTTING DOWN" in caplog.text
