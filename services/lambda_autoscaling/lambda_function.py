"""
Lambda function for ECS Agent Service auto-scaling based on RabbitMQ queue depth.

Polls CloudWatch for queue depth and adjusts ECS service desired count.
Loads configuration from aws.env and .env files.
"""

import os
import json
import math
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from dotenv import dotenv_values
import boto3
from botocore.exceptions import ClientError
from shared.queue_metrics import QUEUE_DEPTH_METRIC

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ecs = boto3.client('ecs')
cloudwatch = boto3.client('cloudwatch')


def load_config():
    """
    Load configuration from aws.env and .env files.
    
    :return: Dictionary with configuration values
    """
    base_dir = Path(__file__).parent
    aws_env_path = base_dir / "aws.env"
    env_path = base_dir / ".env"
    
    config = {}
    
    if aws_env_path.exists():
        config.update(dotenv_values(str(aws_env_path)))
    
    if env_path.exists():
        config.update(dotenv_values(str(env_path)))
    
    return config


CONFIG = load_config()

ECS_CLUSTER = CONFIG.get('ECS_CLUSTER', 'euglena-cluster')
ECS_SERVICE = CONFIG.get('ECS_SERVICE_NAME', 'euglena-agent')
QUEUE_NAME = CONFIG.get('AGENT_INPUT_QUEUE', 'agent.mandates')
TARGET_MESSAGES_PER_WORKER = int(CONFIG.get('TARGET_MESSAGES_PER_WORKER', '1'))
MIN_WORKERS = max(0, int(CONFIG.get('MIN_WORKERS', '0')))
MAX_WORKERS = int(CONFIG.get('MAX_WORKERS', '11'))
CLOUDWATCH_NAMESPACE = CONFIG.get('CLOUDWATCH_NAMESPACE', QUEUE_DEPTH_METRIC.namespace)


def get_queue_depth() -> Optional[int]:
    """
    Get current RabbitMQ queue depth from CloudWatch.
    Uses shorter time window (2 minutes) since metrics are published every 5 seconds.
    
    :return: Queue depth or None on error
    """
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=2)
        
        logger.info(
            f"Querying CloudWatch for queue depth: "
            f"Namespace={CLOUDWATCH_NAMESPACE}, Metric={QUEUE_DEPTH_METRIC.metric_name}, "
            f"QueueName={QUEUE_NAME}, StartTime={start_time}, EndTime={end_time}"
        )
        
        response = cloudwatch.get_metric_statistics(
            Namespace=CLOUDWATCH_NAMESPACE,
            MetricName=QUEUE_DEPTH_METRIC.metric_name,
            Dimensions=QUEUE_DEPTH_METRIC.dimensions(QUEUE_NAME),
            StartTime=start_time,
            EndTime=end_time,
            Period=60,
            Statistics=['Average', 'Maximum']
        )
        
        datapoints = response.get('Datapoints', [])
        if not datapoints:
            logger.warning(
                f"No datapoints found for {QUEUE_DEPTH_METRIC.metric_name} metric. "
                f"Namespace={CLOUDWATCH_NAMESPACE}, QueueName={QUEUE_NAME}, "
                f"TimeWindow=2min. This may indicate metrics are not being published."
            )
            return None
        
        latest = max(datapoints, key=lambda x: x['Timestamp'])
        queue_depth_avg = int(latest.get('Average', 0))
        queue_depth_max = int(latest.get('Maximum', 0))
        timestamp = latest.get('Timestamp')
        
        logger.info(
            f"Queue depth retrieved: Average={queue_depth_avg}, Maximum={queue_depth_max}, "
            f"Timestamp={timestamp}, TotalDatapoints={len(datapoints)}"
        )
        
        return queue_depth_avg
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_msg = e.response.get('Error', {}).get('Message', str(e))
        logger.error(
            f"CloudWatch API error getting queue depth: Code={error_code}, Message={error_msg}"
        )
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting queue depth: {type(e).__name__}: {e}", exc_info=True)
        return None


def get_current_worker_count() -> Optional[int]:
    """
    Get current desired worker count for ECS service.
    
    :return: Current desired count or None on error
    """
    try:
        response = ecs.describe_services(cluster=ECS_CLUSTER, services=[ECS_SERVICE])
        services = response.get('services', [])
        if not services:
            logger.warning(f"Service not found: {ECS_SERVICE} in cluster {ECS_CLUSTER}")
            return None
        
        desired_count = services[0].get('desiredCount', 0)
        logger.info(f"Current desired: {desired_count}")
        return desired_count
    except Exception as e:
        logger.error(f"Error getting worker count: {e}")
        return None


def calculate_desired_workers(queue_depth: int) -> int:
    """
    Calculate desired worker count based on queue depth.
    
    Policy:
    - Minimum: MIN_WORKERS (enforced to be at least 1)
    - If 0 tasks: MIN_WORKERS
    - Otherwise: ceil(tasks / TARGET_MESSAGES_PER_WORKER)
    - Cap at MAX_WORKERS
    
    :param queue_depth: Current queue depth
    :return: Desired worker count (always >= MIN_WORKERS, which is >= 1)
    """
    min_workers = max(1, MIN_WORKERS)
    
    if queue_depth == 0:
        return min_workers
    
    desired = math.ceil(queue_depth / TARGET_MESSAGES_PER_WORKER)
    return max(min_workers, min(desired, MAX_WORKERS))


def update_service_desired_count(desired_count: int) -> bool:
    """
    Update ECS service desired count.
    
    :param desired_count: New desired count
    :return: True on success, False on error
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
    Main Lambda handler for auto-scaling.
    
    When queue depth is unavailable, uses MIN_WORKERS as fallback for testing purposes.
    
    :param event: Lambda event (ignored)
    :param context: Lambda context (ignored)
    :return: Response dictionary
    """
    logger.info(f"Autoscaling check started. Config: cluster={ECS_CLUSTER}, service={ECS_SERVICE}, namespace={CLOUDWATCH_NAMESPACE}, queue={QUEUE_NAME}")
    
    queue_depth = get_queue_depth()
    if queue_depth is None:
        logger.warning(f"Could not get queue depth from CloudWatch. Namespace={CLOUDWATCH_NAMESPACE}, Metric=QueueDepth, QueueName={QUEUE_NAME}")
        logger.info(f"Using fallback: MIN_WORKERS={MIN_WORKERS} (queue depth unavailable)")
        queue_depth = 0
    
    current_count = get_current_worker_count()
    if current_count is None:
        logger.warning(f"Could not get worker count from ECS. Cluster={ECS_CLUSTER}, Service={ECS_SERVICE}")
        return {'statusCode': 200, 'body': json.dumps({'message': 'Could not get worker count', 'action': 'none'})}
    
    desired_count = calculate_desired_workers(queue_depth)
    
    if desired_count == current_count:
        logger.info(f"No scaling: {current_count} workers, queue depth: {queue_depth if queue_depth is not None else 'unavailable (using MIN_WORKERS)'}")
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'No scaling needed',
                'current_count': current_count,
                'desired_count': desired_count,
                'queue_depth': queue_depth if queue_depth is not None else None
            })
        }
    
    success = update_service_desired_count(desired_count)
    action = 'scale_in' if desired_count < current_count else 'scale_out'
    
    if success:
        logger.info(f"Scaling {action}: {current_count} -> {desired_count} (queue: {queue_depth if queue_depth is not None else 'unavailable (using MIN_WORKERS)'})")
        return {
            'statusCode': 200,
            'body': json.dumps({
                'action': action,
                'current_count': current_count,
                'desired_count': desired_count,
                'queue_depth': queue_depth if queue_depth is not None else None
            })
        }
    else:
        return {'statusCode': 500, 'body': json.dumps({'message': 'Failed to update service', 'action': 'error'})}


if __name__ == '__main__':
    result = lambda_handler({}, None)
    print(json.dumps(result, indent=2))
