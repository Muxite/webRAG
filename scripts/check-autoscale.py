"""
Health check script for autoscale deployment mode.
Checks gateway and agent services, ALB target health, and service discovery.
"""
import boto3
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone
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


def check_ecs_services(aws_config: Dict, service_names: List[str], verbose: bool = False):
    """
    Check ECS service health.
    
    :param aws_config: AWS configuration dictionary.
    :param service_names: List of service names to check.
    :param verbose: Whether to show verbose output.
    """
    region = aws_config["AWS_REGION"]
    cluster = aws_config["ECS_CLUSTER"]
    ecs_client = boto3.client("ecs", region_name=region)
    
    print("\n=== ECS Service Status ===")
    
    for service_name in service_names:
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
                    print(f"    Task Definition: {primary.get('taskDefinition', 'UNKNOWN')}")
                
                events = svc.get("events", [])[:3]
                if events:
                    print(f"    Recent Events:")
                    for event in events:
                        print(f"      - {event.get('message', 'N/A')}")
        except Exception as e:
            print(f"  FAIL: {service_name} - {e}")


def check_container_health(aws_config: Dict, service_names: List[str], verbose: bool = False):
    """
    Check container health status.
    
    :param aws_config: AWS configuration dictionary.
    :param service_names: List of service names to check.
    :param verbose: Whether to show verbose output.
    """
    region = aws_config["AWS_REGION"]
    cluster = aws_config["ECS_CLUSTER"]
    ecs_client = boto3.client("ecs", region_name=region)
    
    print("\n=== Container Health ===")
    
    for service_name in service_names:
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
                            "unknown": 0
                        }
                    
                    if health_status == "HEALTHY":
                        container_details[container_name]["healthy"] += 1
                    elif health_status == "UNHEALTHY":
                        container_details[container_name]["unhealthy"] += 1
                    else:
                        container_details[container_name]["unknown"] += 1
            
            print(f"  {service_name}:")
            print(f"    Healthy: {healthy_count}, Unhealthy: {unhealthy_count}, Unknown: {unknown_count}")
            
            if verbose:
                for container_name, details in container_details.items():
                    print(f"    {container_name}: {details['healthy']} healthy, {details['unhealthy']} unhealthy, {details['unknown']} unknown")
        except Exception as e:
            print(f"  FAIL: {service_name} - {e}")


def check_alb_target_health(aws_config: Dict, verbose: bool = False):
    """
    Check ALB target group health for gateway service.
    
    :param aws_config: AWS configuration dictionary.
    :param verbose: Whether to show verbose output.
    """
    region = aws_config["AWS_REGION"]
    elb_client = boto3.client("elbv2", region_name=region)
    
    print("\n=== ALB Target Health ===")
    
    try:
        target_group_arn = aws_config.get("TARGET_GROUP_ARN")
        if not target_group_arn:
            print("  WARN: TARGET_GROUP_ARN not found in aws.env")
            return
        
        response = elb_client.describe_target_health(TargetGroupArn=target_group_arn)
        targets = response.get("TargetHealthDescriptions", [])
        
        if not targets:
            print("  WARN: No targets registered in target group")
            return
        
        healthy_count = 0
        unhealthy_count = 0
        draining_count = 0
        initial_count = 0
        
        for target in targets:
            health = target.get("TargetHealth", {})
            state = health.get("State", "UNKNOWN")
            
            if state == "healthy":
                healthy_count += 1
            elif state == "unhealthy":
                unhealthy_count += 1
            elif state == "draining":
                draining_count += 1
            elif state == "initial":
                initial_count += 1
        
        total = len(targets)
        status = "OK" if (healthy_count > 0 and unhealthy_count == 0) else "FAIL"
        print(f"  {status}: {healthy_count} healthy, {unhealthy_count} unhealthy, {draining_count} draining, {initial_count} initial (total: {total})")
        
        if verbose:
            for target in targets:
                target_id = target.get("Target", {}).get("Id", "unknown")
                port = target.get("Target", {}).get("Port", "unknown")
                health = target.get("TargetHealth", {})
                state = health.get("State", "UNKNOWN")
                reason = health.get("Reason", "N/A")
                description = health.get("Description", "N/A")
                print(f"    Target {target_id}:{port} - {state}")
                if reason != "N/A":
                    print(f"      Reason: {reason}")
                if description != "N/A":
                    print(f"      Description: {description}")
    except Exception as e:
        print(f"  FAIL: ALB target health check - {e}")


def check_service_discovery(aws_config: Dict, verbose: bool = False):
    """
    Check service discovery status for gateway service.
    
    :param aws_config: AWS configuration dictionary.
    :param verbose: Whether to show verbose output.
    """
    region = aws_config["AWS_REGION"]
    sd_client = boto3.client("servicediscovery", region_name=region)
    
    print("\n=== Service Discovery ===")
    
    try:
        namespace_name = aws_config.get("SERVICE_DISCOVERY_NAMESPACE", "euglena.local")
        service_name = "euglena-gateway"
        
        namespaces = sd_client.list_namespaces()["Namespaces"]
        namespace = next((ns for ns in namespaces if ns.get("Name") == namespace_name), None)
        
        if not namespace:
            print(f"  WARN: Namespace '{namespace_name}' not found")
            return
        
        namespace_id = namespace["Id"]
        
        services = sd_client.list_services(
            Filters=[{"Name": "NAMESPACE_ID", "Values": [namespace_id]}]
        )["Services"]
        
        gateway_service = next((s for s in services if s.get("Name") == service_name), None)
        
        if not gateway_service:
            print(f"  WARN: Service '{service_name}' not found in namespace")
            return
        
        service_id = gateway_service["Id"]
        instances = sd_client.list_instances(ServiceId=service_id)["Instances"]
        
        print(f"  OK: {service_name} - {len(instances)} instance(s) registered")
        
        if verbose:
            for instance in instances:
                instance_id = instance.get("Id", "unknown")
                attributes = instance.get("Attributes", {})
                ip = attributes.get("AWS_INSTANCE_IPV4", "N/A")
                port = attributes.get("AWS_INSTANCE_PORT", "N/A")
                print(f"    Instance {instance_id}: {ip}:{port}")
    except Exception as e:
        print(f"  FAIL: Service discovery check - {e}")


def check_stopped_tasks(aws_config: Dict, service_names: List[str], max_tasks: int = 5):
    """
    Check recently stopped tasks and their reasons.
    
    :param aws_config: AWS configuration dictionary.
    :param service_names: List of service names to check.
    :param max_tasks: Maximum number of stopped tasks to show per service.
    """
    region = aws_config["AWS_REGION"]
    cluster = aws_config["ECS_CLUSTER"]
    ecs_client = boto3.client("ecs", region_name=region)
    
    print("\n=== Recently Stopped Tasks ===")
    
    for service_name in service_names:
        try:
            response = ecs_client.list_tasks(
                cluster=cluster,
                serviceName=service_name,
                desiredStatus="STOPPED",
                maxResults=max_tasks
            )
            task_arns = response.get("taskArns", [])
            
            if not task_arns:
                print(f"  {service_name}: No stopped tasks found")
                continue
            
            tasks_response = ecs_client.describe_tasks(cluster=cluster, tasks=task_arns)
            tasks = tasks_response.get("tasks", [])
            
            print(f"  {service_name}: {len(tasks)} stopped task(s)")
            
            for task in tasks[:max_tasks]:
                task_id = task.get("taskArn", "").split("/")[-1]
                stopped_reason = task.get("stoppedReason", "N/A")
                stop_code = task.get("stopCode", "N/A")
                stopped_at = task.get("stoppedAt")
                
                if stopped_at:
                    stopped_time = datetime.fromtimestamp(stopped_at.timestamp())
                    time_ago = datetime.now() - stopped_time
                    time_str = f"{stopped_time.strftime('%Y-%m-%d %H:%M:%S')} ({time_ago.total_seconds()/60:.1f} min ago)"
                else:
                    time_str = "N/A"
                
                print(f"    Task {task_id[:8]}...")
                print(f"      Stop Code: {stop_code}")
                print(f"      Reason: {stopped_reason[:100]}")
                print(f"      Stopped: {time_str}")
                
                containers = task.get("containers", [])
                for container in containers:
                    name = container.get("name", "unknown")
                    exit_code = container.get("exitCode")
                    reason = container.get("reason", "N/A")
                    if exit_code is not None or reason != "N/A":
                        print(f"      Container {name}: exit_code={exit_code}, reason={reason}")
        except Exception as e:
            print(f"  FAIL: {service_name} - {e}")


def check_cloudwatch_logs(aws_config: Dict, service_names: List[str], minutes: int = 10, max_lines: int = 20):
    """
    Check recent CloudWatch logs for errors and warnings.
    
    :param aws_config: AWS configuration dictionary.
    :param service_names: List of service names to check.
    :param minutes: Number of minutes to look back.
    :param max_lines: Maximum number of log lines to show.
    """
    region = aws_config["AWS_REGION"]
    logs_client = boto3.client("logs", region_name=region)
    
    print(f"\n=== CloudWatch Logs (Last {minutes} minutes) ===")
    
    log_group = "/ecs/euglena"
    start_time = int((datetime.now(timezone.utc) - timedelta(minutes=minutes)).timestamp() * 1000)
    
    try:
        for service_name in service_names:
            prefix = service_name.replace("euglena-", "")
            
            try:
                response = logs_client.filter_log_events(
                    logGroupName=log_group,
                    logStreamNamePrefix=prefix,
                    startTime=start_time,
                    filterPattern="ERROR WARN error warn exception Exception failed Failed"
                )
                
                events = response.get("events", [])
                if not events:
                    print(f"  {service_name}: No errors/warnings found")
                    continue
                
                print(f"  {service_name}: Found {len(events)} error/warning event(s)")
                
                for event in events[:max_lines]:
                    timestamp = datetime.fromtimestamp(event.get("timestamp", 0) / 1000)
                    message = event.get("message", "").strip()
                    if message:
                        print(f"    [{timestamp.strftime('%H:%M:%S')}] {message[:150]}")
            except logs_client.exceptions.ResourceNotFoundException:
                print(f"  {service_name}: Log group not found")
            except Exception as e:
                print(f"  {service_name}: Error - {e}")
    except Exception as e:
        print(f"  FAIL: CloudWatch logs check - {e}")


def check_task_definitions(aws_config: Dict, service_names: List[str]):
    """
    Check task definition details.
    
    :param aws_config: AWS configuration dictionary.
    :param service_names: List of service names to check.
    """
    region = aws_config["AWS_REGION"]
    cluster = aws_config["ECS_CLUSTER"]
    ecs_client = boto3.client("ecs", region_name=region)
    
    print("\n=== Task Definitions ===")
    
    for service_name in service_names:
        try:
            response = ecs_client.describe_services(cluster=cluster, services=[service_name])
            services_list = response.get("services", [])
            
            if not services_list:
                continue
            
            svc = services_list[0]
            deployments = svc.get("deployments", [])
            primary = next((d for d in deployments if d.get("status") == "PRIMARY"), None)
            
            if not primary:
                print(f"  {service_name}: No active deployment")
                continue
            
            task_def_arn = primary.get("taskDefinition", "")
            task_def_name = task_def_arn.split("/")[-1].split(":")[0]
            task_def_revision = task_def_arn.split(":")[-1]
            
            task_def_response = ecs_client.describe_task_definition(taskDefinition=task_def_arn)
            task_def = task_def_response.get("taskDefinition", {})
            containers = task_def.get("containerDefinitions", [])
            
            print(f"  {service_name}: {task_def_name}:{task_def_revision}")
            print(f"    CPU: {task_def.get('cpu', 'N/A')}, Memory: {task_def.get('memory', 'N/A')}")
            print(f"    Containers ({len(containers)}):")
            
            for container in containers:
                name = container.get("name", "unknown")
                image = container.get("image", "N/A")
                cpu = container.get("cpu", 0)
                memory = container.get("memory", 0)
                health_check = container.get("healthCheck")
                has_health = "Yes" if health_check else "No"
                print(f"      - {name}: {image.split('/')[-1]}")
                print(f"        CPU: {cpu}, Memory: {memory}, Health Check: {has_health}")
        except Exception as e:
            print(f"  FAIL: {service_name} - {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check autoscale deployment health")
    parser.add_argument("--service", choices=["gateway", "agent", "all"], default="all",
                       help="Service to check (default: all)")
    parser.add_argument("--verbose", action="store_true",
                       help="Show verbose output")
    parser.add_argument("--logs", action="store_true",
                       help="Show CloudWatch logs")
    parser.add_argument("--logs-minutes", type=int, default=10,
                       help="Minutes of logs to show (default: 10)")
    parser.add_argument("--history", action="store_true",
                       help="Show stopped task history")
    parser.add_argument("--task-defs", action="store_true",
                       help="Show task definition details")
    
    args = parser.parse_args()
    
    services_dir = Path.cwd()
    if not (services_dir / "aws.env").exists():
        print(f"Error: aws.env not found in current directory ({services_dir})")
        print("  Please run this script from the services/ directory")
        sys.exit(1)
    
    aws_config = load_aws_config(services_dir)
    
    if args.service == "all":
        service_names = ["euglena-gateway", "euglena-agent"]
    else:
        service_names = [f"euglena-{args.service}"]
    
    print("=" * 60)
    print("Euglena Autoscale Health Check")
    print("=" * 60)
    print(f"Services: {', '.join(service_names)}")
    print(f"Region: {aws_config['AWS_REGION']}")
    print(f"Cluster: {aws_config['ECS_CLUSTER']}")
    print("=" * 60)
    
    check_ecs_services(aws_config, service_names, args.verbose)
    check_container_health(aws_config, service_names, args.verbose)
    
    if "euglena-gateway" in service_names or args.service == "all":
        check_alb_target_health(aws_config, args.verbose)
        check_service_discovery(aws_config, args.verbose)
    
    if args.history:
        check_stopped_tasks(aws_config, service_names)
    
    if args.task_defs:
        check_task_definitions(aws_config, service_names)
    
    if args.logs:
        check_cloudwatch_logs(aws_config, service_names, minutes=args.logs_minutes)
    
    print("\n" + "=" * 60)
    print("Health check complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
