"""
Lambda function for ECS agent autoscaling based on RabbitMQ queue depth.

Polls CloudWatch for queue depth and adjusts ECS service desired count.
Loads configuration from aws.env, .env, and environment variables.
"""

import json
import logging
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from dotenv import dotenv_values
from redis import Redis

from shared.queue_metrics import QUEUE_DEPTH_METRIC

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ecs = boto3.client("ecs")
cloudwatch = boto3.client("cloudwatch")


def load_config() -> dict:
    """
    Load configuration from aws.env, .env, and environment variables.

    :returns: Dictionary with configuration values
    """
    base_dir = Path(__file__).parent
    aws_env_path = base_dir / "aws.env"
    env_path = base_dir / ".env"

    config: dict = {}

    if aws_env_path.exists():
        config.update(dotenv_values(str(aws_env_path)))

    if env_path.exists():
        config.update(dotenv_values(str(env_path)))

    config.update(os.environ)

    return config


CONFIG = load_config()

ECS_CLUSTER = CONFIG.get("ECS_CLUSTER", CONFIG.get("ECS_CLUSTER_NAME", "euglena-cluster"))
ECS_SERVICE = CONFIG.get("AGENT_SERVICE_NAME", CONFIG.get("ECS_SERVICE_NAME", "euglena-agent"))
QUEUE_NAME = CONFIG.get("AGENT_INPUT_QUEUE", "agent.mandates")
TARGET_MESSAGES_PER_WORKER = int(CONFIG.get("TARGET_MESSAGES_PER_WORKER", "2"))
MIN_WORKERS = max(1, int(CONFIG.get("MIN_WORKERS", "1")))
MAX_WORKERS = int(CONFIG.get("MAX_WORKERS", "11"))
CLOUDWATCH_NAMESPACE = CONFIG.get("CLOUDWATCH_NAMESPACE", QUEUE_DEPTH_METRIC.namespace)
REDIS_URL = CONFIG.get("REDIS_URL")
WORKER_STATE_PREFIX = CONFIG.get("WORKER_STATE_PREFIX", "worker_state:agent:")


def get_queue_depth() -> Optional[int]:
    """
    Get current RabbitMQ queue depth from CloudWatch.

    :returns: Queue depth or None on error
    """
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=2)

        logger.info(
            "Querying CloudWatch queue depth: "
            f"namespace={CLOUDWATCH_NAMESPACE}, metric={QUEUE_DEPTH_METRIC.metric_name}, "
            f"queue={QUEUE_NAME}, start={start_time}, end={end_time}"
        )

        response = cloudwatch.get_metric_statistics(
            Namespace=CLOUDWATCH_NAMESPACE,
            MetricName=QUEUE_DEPTH_METRIC.metric_name,
            Dimensions=QUEUE_DEPTH_METRIC.dimensions(QUEUE_NAME),
            StartTime=start_time,
            EndTime=end_time,
            Period=60,
            Statistics=["Average", "Maximum"],
        )

        datapoints = response.get("Datapoints", [])
        if not datapoints:
            logger.warning(
                "No datapoints found for queue depth. "
                f"namespace={CLOUDWATCH_NAMESPACE}, queue={QUEUE_NAME}, window=2min"
            )
            return None

        latest = max(datapoints, key=lambda x: x["Timestamp"])
        queue_depth_avg = int(latest.get("Average", 0))
        queue_depth_max = int(latest.get("Maximum", 0))
        timestamp = latest.get("Timestamp")

        logger.info(
            "Queue depth retrieved: "
            f"avg={queue_depth_avg}, max={queue_depth_max}, ts={timestamp}"
        )

        return queue_depth_avg
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_msg = e.response.get("Error", {}).get("Message", str(e))
        logger.error(f"CloudWatch error getting queue depth: {error_code} {error_msg}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting queue depth: {type(e).__name__}: {e}", exc_info=True)
        return None


def get_protected_worker_count() -> int:
    """
    Count workers that are in working or waiting state.

    :returns: Number of protected workers
    """
    if not REDIS_URL:
        return 0

    try:
        redis = Redis.from_url(REDIS_URL, decode_responses=True)
        cursor = 0
        protected = 0
        pattern = f"{WORKER_STATE_PREFIX}*"
        while True:
            cursor, keys = redis.scan(cursor=cursor, match=pattern, count=200)
            if keys:
                values = redis.mget(keys)
                for value in values:
                    if not value:
                        continue
                    try:
                        payload = json.loads(value)
                    except Exception:
                        continue
                    state = payload.get("state")
                    if state in {"working", "waiting"}:
                        protected += 1
            if cursor == 0:
                break
        return protected
    except Exception as e:
        logger.warning(f"Failed to read worker states: {e}")
        return 0


def get_current_worker_count() -> Optional[int]:
    """
    Get current desired worker count for ECS service.

    :returns: Current desired count or None on error
    """
    try:
        response = ecs.describe_services(cluster=ECS_CLUSTER, services=[ECS_SERVICE])
        services = response.get("services", [])
        if not services:
            logger.warning(f"Service not found: {ECS_SERVICE} in cluster {ECS_CLUSTER}")
            return None

        desired_count = services[0].get("desiredCount", 0)
        logger.info(f"Current desired: {desired_count}")
        return desired_count
    except Exception as e:
        logger.error(f"Error getting worker count: {e}")
        return None


def calculate_desired_workers(queue_depth: int) -> int:
    """
    Calculate desired worker count based on queue depth.

    Policy:
    - Minimum: MIN_WORKERS
    - If 0 tasks: MIN_WORKERS
    - Otherwise: ceil(tasks / TARGET_MESSAGES_PER_WORKER)
    - Cap at MAX_WORKERS

    :param queue_depth: Current queue depth
    :returns: Desired worker count
    """
    if queue_depth == 0:
        return MIN_WORKERS

    desired = math.ceil(queue_depth / TARGET_MESSAGES_PER_WORKER)
    desired = max(MIN_WORKERS, desired)
    if desired > MAX_WORKERS:
        desired = MAX_WORKERS
    return desired


def update_service_desired_count(desired_count: int) -> bool:
    """
    Update ECS service desired count.

    :param desired_count: New desired count
    :returns: True on success, False on error
    """
    try:
        ecs.update_service(cluster=ECS_CLUSTER, service=ECS_SERVICE, desiredCount=desired_count)
        logger.info(f"Updated {ECS_SERVICE} desired count to {desired_count}")
        return True
    except Exception as e:
        logger.error(f"Error updating service: {e}")
        return False


def lambda_handler(event, context):
    """
    Main Lambda handler for autoscaling.

    :param event: Lambda event
    :param context: Lambda context
    :returns: Response dictionary
    """
    logger.info(
        "Autoscaling check started. "
        f"cluster={ECS_CLUSTER}, service={ECS_SERVICE}, "
        f"namespace={CLOUDWATCH_NAMESPACE}, queue={QUEUE_NAME}"
    )

    queue_depth = get_queue_depth()
    if queue_depth is None:
        logger.warning(
            "Queue depth unavailable from CloudWatch. "
            f"namespace={CLOUDWATCH_NAMESPACE}, queue={QUEUE_NAME}"
        )
        queue_depth = 0

    current_count = get_current_worker_count()
    if current_count is None:
        return {"statusCode": 200, "body": json.dumps({"message": "Could not get worker count", "action": "none"})}

    desired_count = calculate_desired_workers(queue_depth)
    protected_workers = get_protected_worker_count()
    if protected_workers > desired_count:
        desired_count = protected_workers

    if desired_count == current_count:
        logger.info(
            f"No scaling: {current_count} workers, queue depth: {queue_depth}, protected: {protected_workers}"
        )
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "No scaling needed",
                    "current_count": current_count,
                    "desired_count": desired_count,
                    "queue_depth": queue_depth,
                    "protected_workers": protected_workers,
                }
            ),
        }

    success = update_service_desired_count(desired_count)
    action = "scale_in" if desired_count < current_count else "scale_out"

    if success:
        logger.info(
            f"Scaling {action}: {current_count} -> {desired_count} "
            f"(queue: {queue_depth}, protected: {protected_workers})"
        )
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "action": action,
                    "current_count": current_count,
                    "desired_count": desired_count,
                    "queue_depth": queue_depth,
                    "protected_workers": protected_workers,
                }
            ),
        }

    return {"statusCode": 500, "body": json.dumps({"message": "Failed to update service", "action": "error"})}


if __name__ == "__main__":
    result = lambda_handler({}, None)
    print(json.dumps(result, indent=2))
