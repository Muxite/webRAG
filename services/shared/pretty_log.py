import logging
import sys
from datetime import datetime


def setup_service_logger(service_name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a standardized logger for a service with enhanced formatting.

    :param service_name: Name of the service ex: Gateway, Agent, Metrics.
    :param level: Logging level default: INFO.
    :returns: Configured logger instance.
    """
    logger = logging.getLogger(service_name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger


def log_connection_status(logger: logging.Logger, service: str, status: str, details: dict = None):
    """
    Log connection status with standardized formatting.

    :param logger: Logger instance.
    :param service: Service name ex: Redis, RabbitMQ.
    :param status: Status ex: CONNECTED, DISCONNECTED, FAILED.
    :param details: Optional dictionary of additional details.
    :returns: None.
    """
    msg = f"{service}: {status}"
    if details:
        detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
        msg = f"{msg}, {detail_str}"
    logger.info(msg)


def log_health_check(logger: logging.Logger, service: str, healthy: bool, components: dict = None):
    """
    Log health check status with standardized formatting.

    :param logger: Logger instance.
    :param service: Service name.
    :param healthy: True if healthy, False otherwise.
    :param components: Optional dictionary of component statuses.
    :returns: None.
    """
    status = "HEALTHY" if healthy else "UNHEALTHY"
    msg = f"{service} Health: {status}"
    if components:
        comp_status = ", ".join(f"{k}={('OK' if v else 'FAIL')}" for k, v in components.items())
        msg = f"{msg}, Components: {comp_status}"
    logger.info(msg)


def pretty_log(data, logger=None, indents=0):
    """
    Writes a well formatted log message to the console with data that has dicts and or lists.
    :param data: Data made of lists and dicts.
    :param logger: Logger to use or None
    :param indents: How many indents to use at the base.
    """
    logger = logger or logging.getLogger(__name__)
    final = pretty_log_print(data, indents)
    logger.info(final)

def pretty_log_graph(graph, logger=None, indents=0):
    """
    Writes a well formatted log message for a graph structure.
    :param graph: Graph instance or dict with nodes/root_id.
    :param logger: Logger to use or None
    :param indents: How many indents to use at the base.
    """
    logger = logger or logging.getLogger(__name__)
    final = pretty_log_graph_print(graph, indents)
    logger.info(final)

def pretty_log_graph_print(graph, indents=0, render: str = "ascii"):
    """
    Format a graph structure with indentation.
    :param graph: Graph instance or dict with nodes/root_id.
    :param indents: How many indents to use at the base.
    :param render: ascii | data
    :returns: Formatted graph string.
    """
    payload = graph.to_dict() if hasattr(graph, "to_dict") else graph
    root_id = payload.get("root_id") if isinstance(payload, dict) else None
    nodes = payload.get("nodes", {}) if isinstance(payload, dict) else {}
    if not root_id or root_id not in nodes:
        return pretty_log_print(payload, indents)

    if render == "data":
        return _build_idea_dag_data(graph)
    if render == "ascii":
        from services.agent.app.idea_dag_log import idea_dag_to_ascii
        return idea_dag_to_ascii(graph)

    return pretty_log_print(payload, indents)

def _build_idea_dag_data(graph):
    """
    Build a graph data structure for idea graph visualization.
    :param graph: Graph instance or dict with nodes.
    :returns: Dict with nodes and edges.
    """
    from services.agent.app.idea_dag_log import idea_dag_data
    return idea_dag_data(graph)

def pretty_log_print(data, indents=0):
    indent_str = "    " * indents
    if isinstance(data, list):
        result = []
        for i, item in enumerate(data):
            result.append(f"{indent_str}{i}. {pretty_log_print(item, indents + 1).lstrip()}")
        return "\n".join(result)
    elif isinstance(data, dict):
        result = []
        for key, value in data.items():
            header = f"{indent_str}{str(key).replace('_', ' ').upper()}:"
            value_str = pretty_log_print(value, indents + 1)
            result.append(f"{header}\n{value_str}")
        return "\n".join(result)
    else:
        return f"{indent_str}{str(data)}"

def main() -> None:
    """
    Run a demo of pretty log rendering.
    :returns: None
    """
    mock_block = {
        "world": "Earth",
        "time": 2025,
        "animals": [
            {"blue_fish": ["tuna", "sardine", "grouper"]},
            {"birds": ["hawk", "sparrow", "pelican"]},
        ],
    }
    print("=== pretty_log_print ===")
    print(pretty_log_print(mock_block, 1))



if __name__ == "__main__":
    main()