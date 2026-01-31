"""
Comprehensive diagnostic script for Euglena ECS deployment.

Gathers extensive data about deployed services including:
- Recent logs from all containers
- Health check status
- Task and service status
- CloudWatch metrics
- EFS mount status
- Network configuration
- Resource utilization
- Error patterns

Expected to be run from services/ directory: python ../scripts/diagnose.py
"""
import boto3
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dotenv import dotenv_values


def load_aws_config(services_dir: Path) -> Dict:
    """
    Load AWS configuration from aws.env file.
    
    :param services_dir: Services directory path.
    :returns: Configuration dictionary.
    """
    aws_env_path = services_dir / "aws.env"
    if not aws_env_path.exists():
        print(f"Error: {aws_env_path} not found")
        sys.exit(1)
    
    return dict(dotenv_values(str(aws_env_path)))


def get_ecs_client(region: str):
    """Get ECS client for the specified region."""
    return boto3.client("ecs", region_name=region)


def get_logs_client(region: str):
    """Get CloudWatch Logs client for the specified region."""
    return boto3.client("logs", region_name=region)


def get_cloudwatch_client(region: str):
    """Get CloudWatch client for the specified region."""
    return boto3.client("cloudwatch", region_name=region)


def get_efs_client(region: str):
    """Get EFS client for the specified region."""
    return boto3.client("efs", region_name=region)


def get_service_info(ecs_client, cluster: str, service_name: str) -> Optional[Dict]:
    """
    Get detailed information about an ECS service.
    
    :param ecs_client: ECS boto3 client.
    :param cluster: ECS cluster name.
    :param service_name: Service name.
    :returns: Service information dictionary or None.
    """
    try:
        response = ecs_client.describe_services(cluster=cluster, services=[service_name])
        services = response.get("services", [])
        if services:
            return services[0]
        return None
    except Exception as e:
        print(f"Error getting service info: {e}")
        return None


def get_running_tasks(ecs_client, cluster: str, service_name: str) -> List[Dict]:
    """
    Get all running tasks for a service.
    
    :param ecs_client: ECS boto3 client.
    :param cluster: ECS cluster name.
    :param service_name: Service name.
    :returns: List of task dictionaries.
    """
    try:
        response = ecs_client.list_tasks(cluster=cluster, serviceName=service_name)
        task_arns = response.get("taskArns", [])
        
        if not task_arns:
            return []
        
        response = ecs_client.describe_tasks(cluster=cluster, tasks=task_arns)
        return response.get("tasks", [])
    except Exception as e:
        print(f"Error getting tasks: {e}")
        return []


def get_task_details(ecs_client, cluster: str, task_arn: str) -> Optional[Dict]:
    """
    Get detailed information about a specific task.
    
    :param ecs_client: ECS boto3 client.
    :param cluster: ECS cluster name.
    :param task_arn: Task ARN.
    :returns: Task details dictionary or None.
    """
    try:
        response = ecs_client.describe_tasks(cluster=cluster, tasks=[task_arn])
        tasks = response.get("tasks", [])
        if tasks:
            return tasks[0]
        return None
    except Exception as e:
        print(f"Error getting task details: {e}")
        return None


def get_container_logs(logs_client, log_group: str, container_name: str, hours: int = 24, limit: int = 100) -> List[Dict]:
    """
    Get recent logs for a container.
    
    :param logs_client: CloudWatch Logs boto3 client.
    :param log_group: Log group name.
    :param container_name: Container name.
    :param hours: Number of hours to look back.
    :param limit: Maximum number of log events to retrieve.
    :returns: List of log event dictionaries.
    """
    try:
        log_stream_prefix = f"{container_name}/"
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(hours=hours)).timestamp() * 1000)
        
        response = logs_client.filter_log_events(
            logGroupName=log_group,
            logStreamNamePrefix=log_stream_prefix,
            startTime=start_time,
            endTime=end_time,
            limit=limit
        )
        
        return response.get("events", [])
    except Exception as e:
        print(f"Error getting logs for {container_name}: {e}")
        return []


def get_error_logs(logs_client, log_group: str, container_name: str, hours: int = 24) -> List[Dict]:
    """
    Get error-level logs for a container.
    
    :param logs_client: CloudWatch Logs boto3 client.
    :param log_group: Log group name.
    :param container_name: Container name.
    :param hours: Number of hours to look back.
    :returns: List of error log event dictionaries.
    """
    try:
        log_stream_prefix = f"{container_name}/"
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(hours=hours)).timestamp() * 1000)
        
        error_keywords = ["error", "ERROR", "Error", "fail", "FAIL", "exception", "Exception", "fatal", "FATAL"]
        
        response = logs_client.filter_log_events(
            logGroupName=log_group,
            logStreamNamePrefix=log_stream_prefix,
            startTime=start_time,
            endTime=end_time,
            filterPattern=" ".join([f'"{kw}"' for kw in error_keywords])
        )
        
        return response.get("events", [])
    except Exception as e:
        print(f"Error getting error logs for {container_name}: {e}")
        return []


def get_cloudwatch_metrics(cloudwatch_client, namespace: str, metric_name: str, dimensions: List[Dict], hours: int = 1) -> List[Dict]:
    """
    Get CloudWatch metrics.
    
    :param cloudwatch_client: CloudWatch boto3 client.
    :param namespace: Metric namespace.
    :param metric_name: Metric name.
    :param dimensions: Metric dimensions.
    :param hours: Number of hours to look back.
    :returns: List of metric data point dictionaries.
    """
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        response = cloudwatch_client.get_metric_statistics(
            Namespace=namespace,
            MetricName=metric_name,
            Dimensions=dimensions,
            StartTime=start_time,
            EndTime=end_time,
            Period=300,
            Statistics=["Average", "Maximum", "Minimum"]
        )
        
        return response.get("Datapoints", [])
    except Exception as e:
        print(f"Error getting metrics {metric_name}: {e}")
        return []


def get_efs_status(efs_client, file_system_id: str) -> Optional[Dict]:
    """
    Get EFS file system status.
    
    :param efs_client: EFS boto3 client.
    :param file_system_id: EFS file system ID.
    :returns: EFS file system information or None.
    """
    try:
        response = efs_client.describe_file_systems(FileSystemId=file_system_id)
        file_systems = response.get("FileSystems", [])
        if file_systems:
            return file_systems[0]
        return None
    except Exception as e:
        print(f"Error getting EFS status: {e}")
        return None


def get_efs_mount_targets(efs_client, file_system_id: str) -> List[Dict]:
    """
    Get EFS mount targets.
    
    :param efs_client: EFS boto3 client.
    :param file_system_id: EFS file system ID.
    :returns: List of mount target dictionaries.
    """
    try:
        response = efs_client.describe_mount_targets(FileSystemId=file_system_id)
        return response.get("MountTargets", [])
    except Exception as e:
        print(f"Error getting EFS mount targets: {e}")
        return []


def format_timestamp(ts: int) -> str:
    """Format Unix timestamp to readable string."""
    return datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M:%S")


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def diagnose_service(aws_config: Dict, service_name: str, hours: int = 24):
    """
    Run comprehensive diagnostics on the ECS service.
    
    :param aws_config: AWS configuration dictionary.
    :param service_name: ECS service name.
    :param hours: Number of hours to look back for logs.
    """
    region = aws_config.get("AWS_REGION", "us-east-2")
    cluster = aws_config.get("ECS_CLUSTER", "euglena-cluster")
    log_group = f"/ecs/{service_name}"
    
    ecs_client = get_ecs_client(region)
    logs_client = get_logs_client(region)
    cloudwatch_client = get_cloudwatch_client(region)
    efs_client = get_efs_client(region)
    
    print_section(f"Euglena Service Diagnostics - {service_name}")
    print(f"Cluster: {cluster}")
    print(f"Region: {region}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print_section("Service Status")
    service_info = get_service_info(ecs_client, cluster, service_name)
    if service_info:
        status = service_info.get('status')
        desired = service_info.get('desiredCount', 0)
        running = service_info.get('runningCount', 0)
        pending = service_info.get('pendingCount', 0)
        
        print(f"Service Name: {service_info.get('serviceName')}")
        print(f"Status: {status}")
        print(f"Desired Count: {desired}")
        print(f"Running Count: {running}")
        print(f"Pending Count: {pending}")
        
        if running == desired and pending == 0:
            print(f"Health: OK")
        elif running < desired or pending > 0:
            print(f"Health: WARN")
        else:
            print(f"Health: FAIL")
        print(f"Task Definition: {service_info.get('taskDefinition', '').split('/')[-1]}")
        print(f"Launch Type: {service_info.get('launchType', 'N/A')}")
        
        deployment_config = service_info.get("deploymentConfiguration", {})
        print(f"Maximum Percent: {deployment_config.get('maximumPercent', 'N/A')}")
        print(f"Minimum Healthy Percent: {deployment_config.get('minimumHealthyPercent', 'N/A')}")
        
        events = service_info.get("events", [])[:10]
        if events:
            print_subsection("Recent Service Events")
            for event in events:
                print(f"  [{event.get('createdAt')}] {event.get('message')}")
    else:
        print("ERROR: Service not found!")
        return
    
    print_section("Running Tasks")
    tasks = get_running_tasks(ecs_client, cluster, service_name)
    print(f"Total Running Tasks: {len(tasks)}")
    
    if not tasks:
        print("WARN: No running tasks found!")
        print("Checking for stopped tasks...")
        
        try:
            response = ecs_client.list_tasks(cluster=cluster, serviceName=service_name, desiredStatus="STOPPED")
            stopped_task_arns = response.get("taskArns", [])[:5]
            
            if stopped_task_arns:
                response = ecs_client.describe_tasks(cluster=cluster, tasks=stopped_task_arns)
                stopped_tasks = response.get("tasks", [])
                
                print(f"Found {len(stopped_tasks)} recently stopped tasks:")
                for task in stopped_tasks:
                    task_arn = task.get("taskArn", "")
                    task_id = task_arn.split("/")[-1]
                    stopped_reason = task.get("stoppedReason", "N/A")
                    stop_code = task.get("stopCode", "N/A")
                    
                    print(f"  Task {task_id}: {stop_code} - {stopped_reason}")
                    
                    containers = task.get("containers", [])
                    for container in containers:
                        container_name = container.get("name", "unknown")
                        exit_code = container.get("exitCode")
                        reason = container.get("reason", "")
                        if exit_code is not None:
                            print(f"    {container_name}: Exit Code {exit_code}, Reason: {reason}")
        except Exception as e:
            print(f"  Error getting stopped tasks: {e}")
        
        return
    
    for i, task in enumerate(tasks, 1):
        task_arn = task.get("taskArn", "")
        task_id = task_arn.split("/")[-1]
        
        print_subsection(f"Task {i}: {task_id}")
        print(f"  ARN: {task_arn}")
        last_status = task.get('lastStatus')
        desired_status = task.get('desiredStatus')
        health_status = task.get('healthStatus', 'N/A')
        
        print(f"  Last Status: {last_status}")
        print(f"  Desired Status: {desired_status}")
        print(f"  Health Status: {health_status}")
        
        if last_status == "RUNNING" and health_status == "HEALTHY":
            print(f"  Status: OK")
        elif last_status == "RUNNING" and health_status != "HEALTHY":
            print(f"  Status: WARN - Running but not healthy")
        elif last_status == "STOPPED":
            print(f"  Status: FAIL - Task stopped")
        else:
            print(f"  Status: WARN - Status: {last_status}")
        print(f"  Started At: {task.get('startedAt', 'N/A')}")
        print(f"  Created At: {task.get('createdAt', 'N/A')}")
        print(f"  Platform Version: {task.get('platformVersion', 'N/A')}")
        print(f"  Platform Family: {task.get('platformFamily', 'N/A')}")
        
        containers = task.get("containers", [])
        print(f"  Containers: {len(containers)}")
        
        for container in containers:
            container_name = container.get("name", "unknown")
            container_status = container.get('lastStatus')
            container_health = container.get('healthStatus', 'N/A')
            exit_code = container.get('exitCode')
            reason = container.get('reason', 'N/A')
            
            print(f"\n    Container: {container_name}")
            print(f"      Status: {container_status}")
            print(f"      Health Status: {container_health}")
            print(f"      Exit Code: {exit_code if exit_code is not None else 'N/A'}")
            print(f"      Reason: {reason}")
            
            if container_status == "RUNNING" and container_health == "HEALTHY":
                print(f"      Health: OK")
            elif container_status == "RUNNING" and container_health != "HEALTHY":
                print(f"      Health: WARN - Running but health check failing")
            elif exit_code is not None:
                if exit_code == 0:
                    print(f"      Health: OK - Exited normally")
                elif exit_code == 137:
                    print(f"      Health: FAIL - Killed (SIGKILL - likely OOM or dependency issue)")
                else:
                    print(f"      Health: FAIL - Exited with code {exit_code}")
            elif container_status == "STOPPED":
                print(f"      Health: FAIL - Container stopped")
            else:
                print(f"      Health: WARN - Status: {container_status}")
            
            cpu = container.get("cpu", "N/A")
            memory = container.get("memory", "N/A")
            print(f"      CPU: {cpu}")
            print(f"      Memory: {memory}")
            
            network_bindings = container.get("networkBindings", [])
            if network_bindings:
                print(f"      Network Bindings:")
                for binding in network_bindings:
                    print(f"        Port {binding.get('containerPort')} -> {binding.get('hostPort')}")
    
    print_section("Container Logs (Recent)")
    container_names = ["gateway", "agent", "rabbitmq", "redis", "chroma"]
    
    for container_name in container_names:
        print_subsection(f"{container_name.upper()} Logs (Last {hours} hours, most recent 50 lines)")
        logs = get_container_logs(logs_client, log_group, container_name, hours=hours, limit=50)
        
        if logs:
            for log in logs[-50:]:
                timestamp = format_timestamp(log.get("timestamp", 0))
                message = log.get("message", "").strip()
                if message:
                    print(f"  [{timestamp}] {message}")
        else:
            print(f"  Status: WARN - No logs found for {container_name}")
            print(f"  This may indicate the container crashed before producing logs")
    
    print_section("Error Logs (Last 24 hours)")
    for container_name in container_names:
        print_subsection(f"{container_name.upper()} Errors")
        error_logs = get_error_logs(logs_client, log_group, container_name, hours=24)
        
        if error_logs:
            print(f"  Status: FAIL - Found {len(error_logs)} error log entries:")
            for log in error_logs[-20:]:
                timestamp = format_timestamp(log.get("timestamp", 0))
                message = log.get("message", "").strip()
                if message:
                    print(f"    [{timestamp}] {message}")
        else:
            print(f"  Status: OK - No errors found for {container_name}")
    
    print_section("CloudWatch Metrics (Last Hour)")
    if tasks:
        task_arn = tasks[0].get("taskArn", "")
        task_id = task_arn.split("/")[-1]
        
        dimensions = [
            {"Name": "ServiceName", "Value": service_name},
            {"Name": "ClusterName", "Value": cluster}
        ]
        
        metrics_to_check = [
            ("AWS/ECS", "CPUUtilization"),
            ("AWS/ECS", "MemoryUtilization"),
            ("AWS/ApplicationELB", "TargetResponseTime"),
            ("AWS/ApplicationELB", "HTTPCode_Target_5XX_Count"),
            ("AWS/ApplicationELB", "HTTPCode_Target_4XX_Count"),
        ]
        
        for namespace, metric_name in metrics_to_check:
            print_subsection(f"{namespace}/{metric_name}")
            datapoints = get_cloudwatch_metrics(cloudwatch_client, namespace, metric_name, dimensions, hours=1)
            if datapoints:
                for dp in sorted(datapoints, key=lambda x: x.get("Timestamp")):
                    ts = dp.get("Timestamp").strftime("%Y-%m-%d %H:%M:%S")
                    avg = dp.get("Average", "N/A")
                    max_val = dp.get("Maximum", "N/A")
                    min_val = dp.get("Minimum", "N/A")
                    print(f"  [{ts}] Avg: {avg}, Max: {max_val}, Min: {min_val}")
            else:
                print(f"  No data available")
    
    print_section("EFS Status")
    efs_file_system_id = aws_config.get("EFS_FILE_SYSTEM_ID", "").strip()
    if efs_file_system_id:
        print(f"EFS File System ID: {efs_file_system_id}")
        
        efs_info = get_efs_status(efs_client, efs_file_system_id)
        if efs_info:
            print(f"  Lifecycle State: {efs_info.get('LifeCycleState')}")
            print(f"  Performance Mode: {efs_info.get('PerformanceMode')}")
            print(f"  Throughput Mode: {efs_info.get('ThroughputMode')}")
            print(f"  Size (bytes): {efs_info.get('SizeInBytes', {}).get('Value', 'N/A')}")
        
        mount_targets = get_efs_mount_targets(efs_client, efs_file_system_id)
        print(f"  Mount Targets: {len(mount_targets)}")
        for mt in mount_targets:
            print(f"    - {mt.get('MountTargetId')} in {mt.get('SubnetId')} (State: {mt.get('LifeCycleState')})")
    else:
        print("  No EFS file system configured")
    
    print_section("Task Definition Details")
    if service_info:
        task_def_arn = service_info.get("taskDefinition", "")
        try:
            response = ecs_client.describe_task_definition(taskDefinition=task_def_arn)
            task_def = response.get("taskDefinition", {})
            
            print(f"Family: {task_def.get('family')}")
            print(f"Revision: {task_def.get('revision')}")
            print(f"Status: {task_def.get('status')}")
            print(f"CPU: {task_def.get('cpu')}")
            print(f"Memory: {task_def.get('memory')}")
            print(f"Network Mode: {task_def.get('networkMode')}")
            
            containers = task_def.get("containerDefinitions", [])
            print(f"\nContainers ({len(containers)}):")
            for container in containers:
                container_name = container.get('name', 'unknown')
                print(f"  - {container_name}")
                print(f"    Image: {container.get('image', 'N/A')}")
                print(f"    CPU: {container.get('cpu', 'N/A')}")
                print(f"    Memory: {container.get('memory', 'N/A')}")
                print(f"    Essential: {container.get('essential', False)}")
                
                health_check = container.get("healthCheck")
                if health_check:
                    command = health_check.get('command', 'N/A')
                    start_period = health_check.get('startPeriod', 'N/A')
                    interval = health_check.get('interval', 'N/A')
                    retries = health_check.get('retries', 'N/A')
                    timeout = health_check.get('timeout', 'N/A')
                    print(f"    Health Check:")
                    print(f"      Command: {command}")
                    print(f"      Start Period: {start_period}s, Interval: {interval}s")
                    print(f"      Retries: {retries}, Timeout: {timeout}s")
                
                depends_on = container.get("dependsOn", [])
                if depends_on:
                    print(f"    Dependencies:")
                    for dep in depends_on:
                        condition = dep.get("condition", "START")
                        dep_name = dep.get("containerName", "unknown")
                        print(f"      - {dep_name} ({condition})")
                
                mount_points = container.get("mountPoints", [])
                if mount_points:
                    print(f"    Mount Points:")
                    for mp in mount_points:
                        print(f"      - {mp.get('sourceVolume')} -> {mp.get('containerPath')}")
                
                env_vars = container.get("environment", [])
                secrets = container.get("secrets", [])
                if env_vars or secrets:
                    print(f"    Environment/Secrets: {len(env_vars)} env vars, {len(secrets)} secrets")
                    critical_vars = ["RABBITMQ_URL", "REDIS_URL", "CHROMA_URL", "GATEWAY_URL"]
                    for var in env_vars:
                        var_name = var.get("name", "")
                        if var_name in critical_vars:
                            var_value = var.get("value", "")
                            if "@" in var_value:
                                var_value = var_value.split("@")[0] + "@***"
                            print(f"      {var_name}: {var_value}")
        except Exception as e:
            print(f"Error getting task definition details: {e}")
    
    print_section("Container Exit Analysis")
    if tasks:
        for task in tasks:
            containers = task.get("containers", [])
            for container in containers:
                container_name = container.get("name", "unknown")
                exit_code = container.get("exitCode")
                reason = container.get("reason", "")
                last_status = container.get("lastStatus", "")
                
                if exit_code is not None or last_status == "STOPPED":
                    print_subsection(f"{container_name.upper()} Container Exit")
                    print(f"  Exit Code: {exit_code}")
                    print(f"  Reason: {reason}")
                    print(f"  Last Status: {last_status}")
                    
                    if exit_code == 1:
                        print(f"  Status: FAIL - Application error (check logs for details)")
                    elif exit_code == 137:
                        print(f"  Status: FAIL - Container killed (SIGKILL - likely OOM or dependency failure)")
                    elif exit_code == 0:
                        print(f"  Status: OK - Normal exit")
                    else:
                        print(f"  Status: FAIL - Unexpected exit code")
    
    print_section("Health Check Analysis")
    if tasks:
        for task in tasks:
            containers = task.get("containers", [])
            for container in containers:
                container_name = container.get("name", "unknown")
                health_status = container.get("healthStatus", "UNKNOWN")
                health_check = container.get("healthCheck", {})
                
                if health_check:
                    print_subsection(f"{container_name.upper()} Health Check")
                    print(f"  Status: {health_status}")
                    if health_status == "HEALTHY":
                        print(f"  Result: OK")
                    elif health_status == "UNHEALTHY":
                        print(f"  Result: FAIL - Health checks failing")
                        print(f"  Command: {health_check.get('command', 'N/A')}")
                    else:
                        print(f"  Result: WARN - Health status unknown")
    
    print_section("Dependency Analysis")
    if service_info:
        task_def_arn = service_info.get("taskDefinition", "")
        try:
            response = ecs_client.describe_task_definition(taskDefinition=task_def_arn)
            task_def = response.get("taskDefinition", {})
            containers = task_def.get("containerDefinitions", [])
            
            print("Container Dependencies:")
            for container in containers:
                container_name = container.get("name", "unknown")
                depends_on = container.get("dependsOn", [])
                if depends_on:
                    print(f"  {container_name} depends on:")
                    for dep in depends_on:
                        condition = dep.get("condition", "START")
                        dep_name = dep.get("containerName", "unknown")
                        print(f"    - {dep_name} ({condition})")
                else:
                    print(f"  {container_name}: No dependencies")
        except Exception as e:
            print(f"Error analyzing dependencies: {e}")
    
    print_section("Resource Utilization")
    if tasks:
        for task in tasks:
            task_arn = task.get("taskArn", "")
            task_id = task_arn.split("/")[-1]
            print_subsection(f"Task {task_id}")
            
            containers = task.get("containers", [])
            total_cpu = 0
            total_memory = 0
            
            for container in containers:
                container_name = container.get("name", "unknown")
                cpu = container.get("cpu", 0)
                memory = container.get("memory", 0)
                
                if isinstance(cpu, str):
                    cpu = int(cpu) if cpu.isdigit() else 0
                if isinstance(memory, str):
                    memory = int(memory) if memory.isdigit() else 0
                
                total_cpu += cpu
                total_memory += memory
                
                print(f"  {container_name}: CPU={cpu}, Memory={memory}MB")
            
            print(f"  Total: CPU={total_cpu}, Memory={total_memory}MB")
            
            task_def_arn = task.get("taskDefinitionArn", "")
            try:
                response = ecs_client.describe_task_definition(taskDefinition=task_def_arn)
                task_def = response.get("taskDefinition", {})
                task_cpu = task_def.get("cpu", "N/A")
                task_memory = task_def.get("memory", "N/A")
                print(f"  Task Limits: CPU={task_cpu}, Memory={task_memory}MB")
            except Exception:
                pass
    
    print_section("Recent Task Failures")
    try:
        response = ecs_client.list_tasks(cluster=cluster, serviceName=service_name, desiredStatus="STOPPED")
        stopped_task_arns = response.get("taskArns", [])[:10]
        
        if stopped_task_arns:
            response = ecs_client.describe_tasks(cluster=cluster, tasks=stopped_task_arns)
            stopped_tasks = response.get("tasks", [])
            
            for task in stopped_tasks:
                task_arn = task.get("taskArn", "")
                task_id = task_arn.split("/")[-1]
                stopped_reason = task.get("stoppedReason", "N/A")
                stop_code = task.get("stopCode", "N/A")
                stopped_at = task.get("stoppedAt", "N/A")
                
                print_subsection(f"Stopped Task {task_id}")
                print(f"  Stopped At: {stopped_at}")
                print(f"  Stop Code: {stop_code}")
                print(f"  Reason: {stopped_reason}")
                
                containers = task.get("containers", [])
                for container in containers:
                    container_name = container.get("name", "unknown")
                    exit_code = container.get("exitCode")
                    reason = container.get("reason", "")
                    if exit_code is not None:
                        print(f"  {container_name}: Exit Code {exit_code}, Reason: {reason}")
        else:
            print("  No recently stopped tasks found")
    except Exception as e:
        print(f"  Error getting stopped tasks: {e}")
    
    print_section("Summary")
    if service_info:
        running = service_info.get("runningCount", 0)
        desired = service_info.get("desiredCount", 0)
        pending = service_info.get("pendingCount", 0)
        
        if running == desired and running > 0 and pending == 0:
            print("Status: OK - Service is healthy - all desired tasks are running")
        elif running < desired or pending > 0:
            print(f"Status: WARN - Service is degraded - {running}/{desired} tasks running, {pending} pending")
        else:
            print("Status: FAIL - Service is unhealthy - no tasks running")
        
        deployments = service_info.get("deployments", [])
        for deployment in deployments:
            rollout_state = deployment.get("rolloutState", "UNKNOWN")
            if rollout_state != "COMPLETED":
                print(f"Deployment Status: {rollout_state}")
                print(f"  Desired: {deployment.get('desiredCount', 0)}")
                print(f"  Running: {deployment.get('runningCount', 0)}")
                print(f"  Pending: {deployment.get('pendingCount', 0)}")
                print(f"  Failed: {deployment.get('failedTasks', 0)}")
    
    print(f"\nDiagnostics completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive ECS service diagnostics")
    parser.add_argument("--service", help="Service name (default: from aws.env)", default=None)
    parser.add_argument("--hours", type=int, help="Hours of logs to retrieve (default: 24)", default=24)
    parser.add_argument("--services-dir", type=Path, default=None,
                       help="Services directory containing aws.env (defaults to current directory)")
    
    args = parser.parse_args()
    
    services_dir = args.services_dir or Path.cwd()
    aws_config = load_aws_config(services_dir)
    
    service_name = args.service or aws_config.get("ECS_SERVICE_NAME", "euglena-service")
    
    diagnose_service(aws_config, service_name, hours=args.hours)


if __name__ == "__main__":
    main()
