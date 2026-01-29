"""
Comprehensive health check script for Euglena services.

Combines all health checks into a single command:
- ECS service status
- Container health
- Network connectivity
- Agent registration
- Task diagnostics

Usage:
    cd services
    python ../scripts/check.py [--service gateway|agent|all] [--verbose]
"""
import boto3
import sys
import argparse
from pathlib import Path
from typing import Dict
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


def check_ecs_services(aws_config: Dict, services: list, verbose: bool = False):
    """
    Check ECS service health.
    
    :param aws_config: AWS configuration dictionary.
    :param services: List of services to check.
    :param verbose: Whether to show verbose output.
    """
    region = aws_config["AWS_REGION"]
    cluster = aws_config["ECS_CLUSTER"]
    ecs_client = boto3.client("ecs", region_name=region)
    
    print("\n=== ECS Service Status ===")
    
    for service in services:
        service_name = f"euglena-{service}"
        
        try:
            response = ecs_client.describe_services(cluster=cluster, services=[service_name])
            services_list = response.get("services", [])
            
            if not services_list:
                print(f"  FAIL: {service_name} not found")
                continue
            
            svc = services_list[0]
            desired = svc.get("desiredCount", 0)
            running = svc.get("runningCount", 0)
            pending = svc.get("pendingCount", 0)
            
            status = "OK" if (running == desired and pending == 0) else "FAIL"
            print(f"  {status}: {service_name} - {running}/{desired} running, {pending} pending")
            
            if verbose:
                deployments = svc.get("deployments", [])
                primary = next((d for d in deployments if d.get("status") == "PRIMARY"), None)
                if primary:
                    print(f"    Deployment: {primary.get('rolloutState', 'UNKNOWN')}")
        except Exception as e:
            print(f"  FAIL: {service_name} - {e}")


def check_container_health(aws_config: Dict, services: list, verbose: bool = False):
    """
    Check container health status.
    
    :param aws_config: AWS configuration dictionary.
    :param services: List of services to check.
    :param verbose: Whether to show verbose output.
    """
    region = aws_config["AWS_REGION"]
    cluster = aws_config["ECS_CLUSTER"]
    ecs_client = boto3.client("ecs", region_name=region)
    
    print("\n=== Container Health ===")
    
    for service in services:
        service_name = f"euglena-{service}"
        
        try:
            response = ecs_client.list_tasks(cluster=cluster, serviceName=service_name)
            task_arns = response.get("taskArns", [])
            
            if not task_arns:
                print(f"  WARN: {service_name} - No tasks running")
                continue
            
            tasks_response = ecs_client.describe_tasks(cluster=cluster, tasks=task_arns[:5])
            tasks = tasks_response.get("tasks", [])
            
            healthy_count = 0
            unhealthy_count = 0
            unknown_count = 0
            container_details = {}
            
            for task in tasks:
                containers = task.get("containers", [])
                for container in containers:
                    container_name = container.get("name", "unknown")
                    health_status = container.get("healthStatus", "UNKNOWN")
                    last_status = container.get("lastStatus", "UNKNOWN")
                    exit_code = container.get("exitCode")
                    health_check = container.get("healthCheck", {})
                    health_reason = health_check.get("reason", "")
                    stopped_reason = container.get("stoppedReason", "")
                    stop_code = container.get("stopCode", "")
                    
                    if health_status == "HEALTHY":
                        healthy_count += 1
                    elif health_status == "UNHEALTHY":
                        unhealthy_count += 1
                    else:
                        unknown_count += 1
                    
                    if container_name not in container_details:
                        container_details[container_name] = {
                            "healthy": 0,
                            "unhealthy": 0,
                            "unknown": 0,
                            "last_status": last_status,
                            "health_reason": "",
                            "exit_code": None,
                            "stopped_reason": "",
                            "stop_code": ""
                        }
                    
                    if health_status == "HEALTHY":
                        container_details[container_name]["healthy"] += 1
                    elif health_status == "UNHEALTHY":
                        container_details[container_name]["unhealthy"] += 1
                    else:
                        container_details[container_name]["unknown"] += 1
                    
                    if health_reason and health_status != "HEALTHY":
                        container_details[container_name]["health_reason"] = health_reason
                    if exit_code is not None:
                        container_details[container_name]["exit_code"] = exit_code
                    if stopped_reason:
                        container_details[container_name]["stopped_reason"] = stopped_reason
                    if stop_code:
                        container_details[container_name]["stop_code"] = stop_code
            
            total = healthy_count + unhealthy_count + unknown_count
            if total > 0:
                status = "OK" if unhealthy_count == 0 and unknown_count == 0 else "FAIL"
                print(f"  {status}: {service_name} - {healthy_count} healthy, {unhealthy_count} unhealthy, {unknown_count} unknown")
                
                if unhealthy_count > 0 or unknown_count > 0:
                    for container_name, details in container_details.items():
                        if details["unhealthy"] > 0 or details["unknown"] > 0:
                            print(f"    FAIL: {container_name} - {details['unhealthy']} unhealthy, {details['unknown']} unknown")
                            if details.get("health_reason"):
                                print(f"      Reason: {details['health_reason']}")
                            if verbose:
                                check_recent_logs(aws_config, service, container_name, limit=5, verbose=verbose)
            else:
                print(f"  WARN: {service_name} - No container health data")
        except Exception as e:
            print(f"  FAIL: {service_name} - {e}")


def check_rabbitmq_health(aws_config: Dict, verbose: bool = False):
    """
    Check RabbitMQ container health in gateway task.
    
    :param aws_config: AWS configuration dictionary.
    :param verbose: Whether to show verbose output.
    """
    print("\n=== RabbitMQ Container Health ===")
    
    region = aws_config["AWS_REGION"]
    cluster = aws_config["ECS_CLUSTER"]
    ecs_client = boto3.client("ecs", region_name=region)
    
    try:
        service_name = "euglena-gateway"
        response = ecs_client.list_tasks(cluster=cluster, serviceName=service_name)
        task_arns = response.get("taskArns", [])
        
        if not task_arns:
            print(f"  WARN: No gateway tasks running")
            return
        
        tasks_response = ecs_client.describe_tasks(cluster=cluster, tasks=task_arns[:1])
        tasks = tasks_response.get("tasks", [])
        
        for task in tasks:
            containers = task.get("containers", [])
            for container in containers:
                container_name = container.get("name", "")
                if "rabbitmq" in container_name.lower():
                    health_status = container.get("healthStatus", "UNKNOWN")
                    status = "OK" if health_status == "HEALTHY" else "FAIL"
                    print(f"  {status}: RabbitMQ - {health_status}")
                    return
        
        print("  WARN: RabbitMQ container not found")
    except Exception as e:
        print(f"  FAIL: RabbitMQ check - {e}")


def check_network(aws_config: Dict, verbose: bool = False):
    """
    Check network configuration.
    
    :param aws_config: AWS configuration dictionary.
    :param verbose: Whether to show verbose output.
    """
    print("\n=== Network Configuration ===")
    
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from network_utils import validate_network_configuration
        
        is_valid, issues = validate_network_configuration(aws_config)
        
        if is_valid:
            print("  OK: Network configuration valid")
        else:
            print(f"  FAIL: Network validation - {len(issues)} issues")
            for issue in issues[:3]:
                print(f"    - {issue}")
    except Exception as e:
        print(f"  FAIL: Network check - {e}")


def check_recent_logs(aws_config: Dict, service: str, container_name: str, limit: int = 20, verbose: bool = False) -> None:
    """
    Fetch and display recent logs for a specific container.
    
    :param aws_config: AWS configuration dictionary.
    :param service: Service name (gateway or agent).
    :param container_name: Container name to check.
    :param limit: Number of log events to fetch.
    :param verbose: Whether to show verbose output.
    """
    region = aws_config["AWS_REGION"]
    logs_client = boto3.client("logs", region_name=region)
    ecs_client = boto3.client("ecs", region_name=region)
    
    try:
        cluster = aws_config["ECS_CLUSTER"]
        service_name = f"euglena-{service}"
        
        response = ecs_client.list_tasks(cluster=cluster, serviceName=service_name)
        task_arns = response.get("taskArns", [])
        
        if not task_arns:
            return
        
        tasks_response = ecs_client.describe_tasks(cluster=cluster, tasks=task_arns[:1])
        tasks = tasks_response.get("tasks", [])
        
        if not tasks:
            return
        
        task = tasks[0]
        task_id = task.get("taskArn", "").split("/")[-1]
        
        log_group = "/ecs/euglena"
        log_stream_prefix = f"{service}/{container_name}/{task_id}"
        
        streams = logs_client.describe_log_streams(
            logGroupName=log_group,
            logStreamNamePrefix=log_stream_prefix,
            limit=1
        )
        
        if streams.get("logStreams"):
            stream_name = streams["logStreams"][0]["logStreamName"]
            events = logs_client.get_log_events(
                logGroupName=log_group,
                logStreamName=stream_name,
                limit=limit,
                startFromHead=False
            )
            
            print(f"    Recent {container_name} logs:")
            for event in events.get("events", [])[-5:]:
                message = event.get("message", "")
                if any(keyword in message.lower() for keyword in ["error", "failed", "exception"]):
                    print(f"      {message[:150]}")
    except Exception:
        pass


def check_health_endpoints(aws_config: Dict, verbose: bool = False):
    """
    Check if health endpoints are actually responding.
    
    :param aws_config: AWS configuration dictionary.
    :param verbose: Whether to show verbose output.
    """
    print("\n=== Health Endpoint Status ===")
    
    region = aws_config["AWS_REGION"]
    cluster = aws_config["ECS_CLUSTER"]
    ecs_client = boto3.client("ecs", region_name=region)
    
    try:
        for service in ["gateway", "agent"]:
            service_name = f"euglena-{service}"
            port = 8080 if service == "gateway" else 8081
            
            response = ecs_client.list_tasks(cluster=cluster, serviceName=service_name)
            task_arns = response.get("taskArns", [])
            
            if not task_arns:
                continue
            
            tasks_response = ecs_client.describe_tasks(cluster=cluster, tasks=task_arns[:1])
            tasks = tasks_response.get("tasks", [])
            
            if not tasks:
                continue
            
            task = tasks[0]
            containers = task.get("containers", [])
            service_container = next((c for c in containers if c.get("name") == service), None)
            
            if not service_container:
                continue
            
            health_status = service_container.get("healthStatus", "UNKNOWN")
            status = "OK" if health_status == "HEALTHY" else "FAIL"
            print(f"  {status}: {service_name} - {health_status}")
    except Exception as e:
        print(f"  FAIL: Health endpoint check - {e}")


def check_chroma_storage(aws_config: Dict, verbose: bool = False):
    """
    Check if Chroma EFS storage is properly configured.
    
    :param aws_config: AWS configuration dictionary.
    :param verbose: Whether to show verbose output.
    """
    print("\n=== Chroma Storage Configuration ===")
    
    region = aws_config["AWS_REGION"]
    cluster = aws_config["ECS_CLUSTER"]
    efs_client = boto3.client("efs", region_name=region)
    ecs_client = boto3.client("ecs", region_name=region)
    
    chroma_fs_id = aws_config.get("CHROMA_EFS_FILE_SYSTEM_ID", "").strip()
    
    if not chroma_fs_id:
        print("  WARN: CHROMA_EFS_FILE_SYSTEM_ID not set - using ephemeral storage")
        return
    
    print(f"  OK: CHROMA_EFS_FILE_SYSTEM_ID configured: {chroma_fs_id}")
    
    try:
        efs_response = efs_client.describe_file_systems(FileSystemId=chroma_fs_id)
        file_systems = efs_response.get("FileSystems", [])
        
        if not file_systems:
            print(f"  FAIL: EFS file system {chroma_fs_id} not found")
            return
        
        fs = file_systems[0]
        fs_status = fs.get("LifeCycleState", "UNKNOWN")
        
        if fs_status == "available":
            print(f"  OK: EFS file system status - {fs_status}")
        else:
            print(f"  FAIL: EFS file system status - {fs_status}")
            return
        
        mount_targets = efs_client.describe_mount_targets(FileSystemId=chroma_fs_id)
        targets = mount_targets.get("MountTargets", [])
        
        if targets:
            available_targets = [t for t in targets if t.get("LifeCycleState") == "available"]
            status = "OK" if len(available_targets) > 0 else "FAIL"
            print(f"  {status}: Mount targets - {len(available_targets)}/{len(targets)} available")
        else:
            print(f"  WARN: No mount targets found")
    
    except Exception as e:
        error_code = getattr(e, 'response', {}).get('Error', {}).get('Code', '')
        if error_code == 'FileSystemNotFound':
            print(f"  FAIL: EFS file system {chroma_fs_id} not found")
        else:
            print(f"  FAIL: EFS check - {e}")
    
    try:
        service_name = "euglena-gateway"
        response = ecs_client.describe_services(cluster=cluster, services=[service_name])
        services_list = response.get("services", [])
        
        if not services_list:
            return
        
        svc = services_list[0]
        task_def_arn = svc.get("taskDefinition", "")
        
        if not task_def_arn:
            return
        
        task_def_response = ecs_client.describe_task_definition(taskDefinition=task_def_arn)
        task_def = task_def_response.get("taskDefinition", {})
        
        volumes = task_def.get("volumes", [])
        chroma_volume = next((v for v in volumes if v.get("name") == "chroma-volume"), None)
        
        if chroma_volume:
            efs_config = chroma_volume.get("efsVolumeConfiguration", {})
            configured_fs_id = efs_config.get("fileSystemId", "")
            
            if configured_fs_id == chroma_fs_id:
                print(f"  OK: Task definition EFS volume configured")
            else:
                print(f"  FAIL: Task definition EFS volume mismatch - expected {chroma_fs_id}, found {configured_fs_id}")
        else:
            print(f"  FAIL: Task definition missing chroma-volume")
        
        container_defs = task_def.get("containerDefinitions", [])
        chroma_container = next((c for c in container_defs if c.get("name") == "chroma"), None)
        
        if chroma_container:
            mount_points = chroma_container.get("mountPoints", [])
            chroma_mount = next((m for m in mount_points if m.get("sourceVolume") == "chroma-volume"), None)
            
            if chroma_mount and chroma_mount.get("containerPath") == "/chroma-data":
                print(f"  OK: Chroma container mount point configured")
            else:
                print(f"  FAIL: Chroma container mount point missing or incorrect")
    
    except Exception as e:
        print(f"  FAIL: Task definition check - {e}")


def check_agent_registration(aws_config: Dict, verbose: bool = False):
    """
    Check if agent is registered in Redis and diagnose connection issues.
    
    :param aws_config: AWS configuration dictionary.
    :param verbose: Whether to show verbose output.
    """
    print("\n=== Agent Registration & Connection Status ===")
    
    region = aws_config["AWS_REGION"]
    logs_client = boto3.client("logs", region_name=region)
    
    try:
        log_group = "/ecs/euglena"
        agent_streams = logs_client.describe_log_streams(
            logGroupName=log_group,
            logStreamNamePrefix="agent",
            limit=10
        )
        
        if agent_streams.get("logStreams"):
            agent_streams["logStreams"].sort(
                key=lambda x: x.get("lastEventTimestamp", 0),
                reverse=True
            )
        
        if agent_streams.get("logStreams"):
            stream_name = agent_streams["logStreams"][0]["logStreamName"]
            events = logs_client.get_log_events(
                logGroupName=log_group,
                logStreamName=stream_name,
                limit=200,
                startFromHead=False
            )
            
            redis_connected = False
            rabbitmq_connected = False
            presence_started = False
            errors_found = []
            
            for event in events.get("events", []):
                message = event.get("message", "")
                timestamp = event.get("timestamp", 0)
                
                if "Redis" in message and ("ready" in message.lower() or "connected" in message.lower()):
                    redis_connected = True
                
                if "RabbitMQ" in message and ("connected" in message.lower() or "CONNECTED" in message):
                    rabbitmq_connected = True
                
                if "Presence" in message or "presence" in message.lower():
                    presence_started = True
                
                if "CONNECTION_FORCED" in message or "shutdown" in message.lower():
                    errors_found.append(message[:100])
                
                if "ChannelInvalidStateError" in message or "No active transport" in message:
                    errors_found.append(message[:100])
            
            redis_status = "OK" if redis_connected else "FAIL"
            rabbitmq_status = "OK" if rabbitmq_connected else "FAIL"
            presence_status = "OK" if presence_started else "FAIL"
            
            print(f"  {redis_status}: Redis connection")
            print(f"  {rabbitmq_status}: RabbitMQ connection")
            print(f"  {presence_status}: Worker presence")
            
            if errors_found and verbose:
                print(f"  Recent errors ({len(errors_found)}):")
                for error in errors_found[-3:]:
                    print(f"    - {error}")
    except Exception as e:
        print(f"  FAIL: Agent registration check - {e}")


def main():
    """
    Main health check entry point.
    
    Expected to be run from services/ directory.
    """
    parser = argparse.ArgumentParser(description="Check Euglena service health")
    parser.add_argument("--service", choices=["gateway", "agent", "all"], default="all",
                       help="Service to check (default: all)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show verbose output")
    
    args = parser.parse_args()
    
    services_dir = Path.cwd()
    if not (services_dir / "aws.env").exists():
        print("Error: Must run from services/ directory")
        sys.exit(1)
    
    aws_config = load_aws_config(services_dir)
    
    services = ["gateway", "agent"] if args.service == "all" else [args.service]
    
    print("=" * 60)
    print("Euglena Health Check")
    print("=" * 60)
    print(f"Services: {', '.join(services)}")
    print(f"Region: {aws_config['AWS_REGION']}")
    print("=" * 60)
    
    check_ecs_services(aws_config, services, args.verbose)
    check_container_health(aws_config, services, args.verbose)
    
    if "gateway" in services or args.service == "all":
        check_rabbitmq_health(aws_config, args.verbose)
    
    check_network(aws_config, args.verbose)
    check_health_endpoints(aws_config, args.verbose)
    
    if "gateway" in services or args.service == "all":
        check_chroma_storage(aws_config, args.verbose)
    
    if "agent" in services:
        check_agent_registration(aws_config, args.verbose)
    
    print("\n" + "=" * 60)
    print("Health check complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
