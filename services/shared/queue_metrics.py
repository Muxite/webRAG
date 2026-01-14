"""
Shared definitions for RabbitMQ queue depth metrics.

:param none: No parameters.
:returns: Constants and helpers used by metric publishers and consumers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueueDepthMetric:
    """
    CloudWatch metric definition for RabbitMQ queue depth.

    :param namespace: CloudWatch namespace (e.g. "Euglena/RabbitMQ").
    :param metric_name: Metric name (e.g. "QueueDepth").
    :param queue_name_dimension: Dimension key for the queue name.
    :returns: Immutable metric definition.
    """

    namespace: str = "Euglena/RabbitMQ"
    metric_name: str = "QueueDepth"
    queue_name_dimension: str = "QueueName"

    def dimensions(self, queue_name: str) -> list[dict]:
        """
        Build CloudWatch dimensions for a specific queue.

        :param queue_name: RabbitMQ queue name.
        :returns: Dimensions list for CloudWatch APIs.
        """

        return [{"Name": self.queue_name_dimension, "Value": queue_name}]


QUEUE_DEPTH_METRIC = QueueDepthMetric()

