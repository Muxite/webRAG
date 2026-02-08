"""
Comprehensive diagnostic script for autoscale deployment.
Gathers data from ECS, CloudWatch, Service Discovery, and other AWS services.
"""
import boto3
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dotenv import dotenv_values

try:
    from scripts.deployment_mode import DeploymentMode
except ImportError:
    from deployment_mode import DeploymentMode


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


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")


def diagnose_ecs_services(aws_config: Dict, service_names: List[str]):
    """
    Diagnose ECS services - status, deployments, events.
    
    :param aws_config: AWS configuration dictionary.
    :param service_names: List of service names to check.
    """
    print_section("ECS Services Status")
    
    region = aws_config["AWS_REGION"]
    cluster = aws_config["ECS_CLUSTER"]
    ecs_client = boto3.client("ecs", region_name=region)
    
    for service_name in service_names:
        print(f"\n--- {service_name} ---")
        try:
            response = ecs_client.describe_services(cluster=cluster, services=[service_name])
            services = response.get("services", [])
            
            if not services:
                print(f"  [FAIL] Service not found")
                continue
            
            svc = services[0]
            print(f"  Status: {svc.get('status', 'UNKNOWN')}")
            print(f"  Desired: {svc.get('desiredCount', 0)}")
            print(f"  Running: {svc.get('runningCount', 0)}")
            print(f"  Pending: {svc.get('pendingCount', 0)}")
            print(f"  Launch Type: {svc.get('launchType', 'UNKNOWN')}")
            print(f"  Platform Version: {svc.get('platformVersion', 'UNKNOWN')}")
            
            deployments = svc.get("deployments", [])
            print(f"\n  Deployments ({len(deployments)}):")
            for dep in deployments:
                print(f"    - ID: {dep.get('id', 'UNKNOWN')}")
                print(f"      Status: {dep.get('status', 'UNKNOWN')}")
                print(f"      Rollout State: {dep.get('rolloutState', 'UNKNOWN')}")
                print(f"      Running: {dep.get('runningCount', 0)}")
                print(f"      Desired: {dep.get('desiredCount', 0)}")
                print(f"      Task Definition: {dep.get('taskDefinition', 'UNKNOWN').split('/')[-1]}")
            
            events = svc.get("events", [])
            print(f"\n  Recent Events ({len(events)}):")
            for event in events[:10]:
                print(f"    [{event.get('createdAt', 'UNKNOWN')}] {event.get('message', '')}")
            
            network_config = svc.get("networkConfiguration", {})
            awsvpc = network_config.get("awsvpcConfiguration", {})
            if awsvpc:
                print(f"\n  Network Configuration:")
                print(f"    Subnets: {', '.join(awsvpc.get('subnets', []))}")
                print(f"    Security Groups: {', '.join(awsvpc.get('securityGroups', []))}")
                print(f"    Assign Public IP: {awsvpc.get('assignPublicIp', 'UNKNOWN')}")
            
            service_registries = svc.get("serviceRegistries", [])
            if service_registries:
                print(f"\n  Service Discovery:")
                for reg in service_registries:
                    print(f"    Registry ARN: {reg.get('registryArn', 'UNKNOWN')}")
                    print(f"    Port: {reg.get('port', 'N/A')}")
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()


def diagnose_tasks(aws_config: Dict, service_names: List[str]):
    """
    Diagnose ECS tasks - status, containers, stop codes, errors.
    
    :param aws_config: AWS configuration dictionary.
    :param service_names: List of service names to check.
    """
    print_section("ECS Tasks Status")
    
    region = aws_config["AWS_REGION"]
    cluster = aws_config["ECS_CLUSTER"]
    ecs_client = boto3.client("ecs", region_name=region)
    
    for service_name in service_names:
        print(f"\n--- {service_name} ---")
        try:
            response = ecs_client.list_tasks(cluster=cluster, serviceName=service_name)
            task_arns = response.get("taskArns", [])
            
            if not task_arns:
                print(f"  [WARN] No tasks found")
                continue
            
            print(f"  Found {len(task_arns)} task(s)")
            
            tasks_response = ecs_client.describe_tasks(cluster=cluster, tasks=task_arns)
            tasks = tasks_response.get("tasks", [])
            
            for task in tasks:
                task_id = task.get("taskArn", "").split("/")[-1]
                print(f"\n  Task: {task_id}")
                print(f"    Last Status: {task.get('lastStatus', 'UNKNOWN')}")
                print(f"    Desired Status: {task.get('desiredStatus', 'UNKNOWN')}")
                print(f"    Health Status: {task.get('healthStatus', 'UNKNOWN')}")
                print(f"    Created At: {task.get('createdAt', 'UNKNOWN')}")
                print(f"    Started At: {task.get('startedAt', 'N/A')}")
                print(f"    Stopped At: {task.get('stoppedAt', 'N/A')}")
                print(f"    Stop Code: {task.get('stopCode', 'N/A')}")
                print(f"    Stopped Reason: {task.get('stoppedReason', 'N/A')}")
                
                containers = task.get("containers", [])
                print(f"    Containers ({len(containers)}):")
                for container in containers:
                    container_name = container.get("name", "unknown")
                    print(f"      - {container_name}:")
                    print(f"        Status: {container.get('lastStatus', 'UNKNOWN')}")
                    print(f"        Health: {container.get('healthStatus', 'UNKNOWN')}")
                    print(f"        Exit Code: {container.get('exitCode', 'N/A')}")
                    reason = container.get("reason", "")
                    if reason:
                        print(f"        Reason: {reason}")
                    
                    stopped_at = container.get("stoppedAt")
                    if stopped_at:
                        print(f"        Stopped At: {stopped_at}")
                
                stopped_tasks_response = ecs_client.describe_tasks(
                    cluster=cluster,
                    tasks=[task.get("taskArn")],
                    include=["TAGS"]
                )
                
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()


def diagnose_stopped_tasks(aws_config: Dict, service_names: List[str], limit: int = 10):
    """
    Diagnose recently stopped tasks to find errors.
    
    :param aws_config: AWS configuration dictionary.
    :param service_names: List of service names to check.
    :param limit: Maximum number of stopped tasks to check.
    """
    print_section("Recently Stopped Tasks (Error Analysis)")
    
    region = aws_config["AWS_REGION"]
    cluster = aws_config["ECS_CLUSTER"]
    ecs_client = boto3.client("ecs", region_name=region)
    
    for service_name in service_names:
        print(f"\n--- {service_name} ---")
        try:
            response = ecs_client.list_tasks(
                cluster=cluster,
                serviceName=service_name,
                desiredStatus="STOPPED"
            )
            task_arns = response.get("taskArns", [])
            
            if not task_arns:
                print(f"  [OK] No stopped tasks found")
                continue
            
            print(f"  Found {len(task_arns)} stopped task(s), checking most recent {limit}...")
            
            tasks_response = ecs_client.describe_tasks(
                cluster=cluster,
                tasks=task_arns[:limit]
            )
            tasks = tasks_response.get("tasks", [])
            
            for task in tasks:
                task_id = task.get("taskArn", "").split("/")[-1]
                stop_code = task.get("stopCode", "UNKNOWN")
                stopped_reason = task.get("stoppedReason", "")
                stopped_at = task.get("stoppedAt")
                
                print(f"\n  Task: {task_id}")
                print(f"    Stop Code: {stop_code}")
                print(f"    Stopped At: {stopped_at}")
                print(f"    Stopped Reason: {stopped_reason}")
                
                containers = task.get("containers", [])
                for container in containers:
                    container_name = container.get("name", "unknown")
                    exit_code = container.get("exitCode")
                    reason = container.get("reason", "")
                    print(f"    Container '{container_name}':")
                    print(f"      Exit Code: {exit_code}")
                    if reason:
                        print(f"      Reason: {reason}")
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()


def diagnose_task_definitions(aws_config: Dict, task_families: List[str]):
    """
    Diagnose task definitions - check latest revisions.
    
    :param aws_config: AWS configuration dictionary.
    :param task_families: List of task definition families to check.
    """
    print_section("Task Definitions")
    
    region = aws_config["AWS_REGION"]
    ecs_client = boto3.client("ecs", region_name=region)
    
    for family in task_families:
        print(f"\n--- {family} ---")
        try:
            response = ecs_client.list_task_definitions(familyPrefix=family, status="ACTIVE", sort="DESC")
            task_def_arns = response.get("taskDefinitionArns", [])
            
            if not task_def_arns:
                print(f"  [WARN] No task definitions found")
                continue
            
            latest_arn = task_def_arns[0]
            print(f"  Latest: {latest_arn.split('/')[-1]}")
            
            task_def_response = ecs_client.describe_task_definition(taskDefinition=latest_arn)
            task_def = task_def_response.get("taskDefinition", {})
            
            containers = task_def.get("containerDefinitions", [])
            print(f"  Containers ({len(containers)}):")
            for container in containers:
                container_name = container.get("name", "unknown")
                image = container.get("image", "UNKNOWN")
                print(f"    - {container_name}:")
                print(f"      Image: {image}")
                print(f"      CPU: {container.get('cpu', 0)}")
                print(f"      Memory: {container.get('memory', 0)}")
                print(f"      Memory Reservation: {container.get('memoryReservation', 'N/A')}")
                
                health_check = container.get("healthCheck")
                if health_check:
                    print(f"      Health Check: {health_check.get('command', [])}")
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()


def diagnose_cloudwatch_logs(aws_config: Dict, log_groups: List[str], hours: int = 1):
    """
    Diagnose CloudWatch logs for recent errors.
    
    :param aws_config: AWS configuration dictionary.
    :param log_groups: List of log group names to check.
    :param hours: Number of hours to look back.
    """
    print_section("CloudWatch Logs (Recent Errors)")
    
    region = aws_config["AWS_REGION"]
    logs_client = boto3.client("logs", region_name=region)
    
    start_time = int((datetime.utcnow() - timedelta(hours=hours)).timestamp() * 1000)
    
    for log_group in log_groups:
        print(f"\n--- {log_group} ---")
        try:
            response = logs_client.filter_log_events(
                logGroupName=log_group,
                startTime=start_time,
                filterPattern="ERROR error Error exception Exception failed Failed FAIL"
            )
            
            events = response.get("events", [])
            if not events:
                print(f"  [OK] No errors found in last {hours} hour(s)")
            else:
                print(f"  [WARN] Found {len(events)} error event(s):")
                for event in events[:20]:
                    timestamp = datetime.fromtimestamp(event.get("timestamp", 0) / 1000)
                    message = event.get("message", "")
                    print(f"    [{timestamp}] {message[:200]}")
        except logs_client.exceptions.ResourceNotFoundException:
            print(f"  [WARN] Log group not found")
        except Exception as e:
            print(f"  [ERROR] {e}")


def diagnose_service_discovery(aws_config: Dict):
    """
    Diagnose AWS Cloud Map service discovery configuration.
    
    :param aws_config: AWS configuration dictionary.
    """
    print_section("Service Discovery")
    
    region = aws_config["AWS_REGION"]
    namespace_name = aws_config.get("SERVICE_DISCOVERY_NAMESPACE", "euglena.local")
    service_name = "euglena-gateway"
    
    servicediscovery_client = boto3.client("servicediscovery", region_name=region)
    
    try:
        response = servicediscovery_client.list_namespaces()
        namespaces = response.get("Namespaces", [])
        
        namespace_id = None
        for namespace in namespaces:
            if namespace.get("Name") == namespace_name:
                namespace_id = namespace.get("Id")
                print(f"  Namespace: {namespace_name} (ID: {namespace_id})")
                print(f"    Type: {namespace.get('Type', 'UNKNOWN')}")
                print(f"    ARN: {namespace.get('Arn', 'UNKNOWN')}")
                break
        
        if not namespace_id:
            print(f"  [WARN] Namespace '{namespace_name}' not found")
            return
        
        response = servicediscovery_client.list_services(
            Filters=[{"Name": "NAMESPACE_ID", "Values": [namespace_id]}]
        )
        services = response.get("Services", [])
        
        for svc in services:
            if svc.get("Name") == service_name:
                service_id = svc.get("Id")
                print(f"\n  Service: {service_name} (ID: {service_id})")
                print(f"    ARN: {svc.get('Arn', 'UNKNOWN')}")
                
                service_detail = servicediscovery_client.get_service(Id=service_id)
                service_data = service_detail.get("Service", {})
                
                dns_config = service_data.get("DnsConfig", {})
                if dns_config:
                    dns_records = dns_config.get("DnsRecords", [])
                    print(f"    DNS Records:")
                    for record in dns_records:
                        print(f"      Type: {record.get('Type')}, TTL: {record.get('TTL')}")
                
                health_check = service_data.get("HealthCheckConfig", {})
                if health_check:
                    print(f"    Health Check:")
                    print(f"      Type: {health_check.get('Type')}")
                    print(f"      Resource Path: {health_check.get('ResourcePath', 'N/A')}")
                    print(f"      Failure Threshold: {health_check.get('FailureThreshold', 'N/A')}")
                break
        else:
            print(f"  [WARN] Service '{service_name}' not found in namespace")
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()


def diagnose_ecr_images(aws_config: Dict, repositories: List[str]):
    """
    Diagnose ECR repositories and images.
    
    :param aws_config: AWS configuration dictionary.
    :param repositories: List of repository names to check.
    """
    print_section("ECR Images")
    
    region = aws_config["AWS_REGION"]
    account_id = aws_config["AWS_ACCOUNT_ID"]
    ecr_client = boto3.client("ecr", region_name=region)
    
    for repo_name in repositories:
        full_name = f"euglena/{repo_name}"
        print(f"\n--- {full_name} ---")
        try:
            response = ecr_client.describe_repositories(repositoryNames=[full_name])
            repos = response.get("repositories", [])
            
            if not repos:
                print(f"  [WARN] Repository not found")
                continue
            
            repo = repos[0]
            print(f"  URI: {repo.get('repositoryUri', 'UNKNOWN')}")
            print(f"  Created: {repo.get('createdAt', 'UNKNOWN')}")
            
            images_response = ecr_client.describe_images(repositoryName=full_name, maxResults=5)
            images = images_response.get("imageDetails", [])
            
            if not images:
                print(f"  [WARN] No images found")
                continue
            
            print(f"  Images ({len(images)}):")
            for image in images:
                tags = image.get("imageTags", ["<untagged>"])
                pushed_at = image.get("imagePushedAt", "UNKNOWN")
                size = image.get("imageSizeInBytes", 0)
                size_mb = size / (1024 * 1024) if size else 0
                print(f"    - Tags: {', '.join(tags)}")
                print(f"      Pushed: {pushed_at}")
                print(f"      Size: {size_mb:.2f} MB")
        except ecr_client.exceptions.RepositoryNotFoundException:
            print(f"  [WARN] Repository not found")
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()


def parse_args():
    """
    Parse CLI arguments.

    :returns: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Comprehensive deployment diagnostics")
    parser.add_argument("--mode", choices=["single", "autoscale"], default="autoscale",
                       help="Deployment mode to diagnose")
    parser.add_argument("--hours", type=int, default=1,
                       help="Hours of logs to check (default: 1)")
    parser.add_argument("--skip-logs", action="store_true",
                       help="Skip CloudWatch logs check")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    mode = DeploymentMode.from_string(args.mode)
    
    services_dir = Path.cwd()
    if not (services_dir / "aws.env").exists():
        print(f"Error: aws.env not found in current directory ({services_dir})")
        print("  Please run this script from the services/ directory")
        sys.exit(1)
    
    aws_config = load_aws_config(services_dir)
    
    print("=" * 80)
    print(" Euglena Deployment Diagnostics")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Region: {aws_config['AWS_REGION']}")
    print(f"Cluster: {aws_config['ECS_CLUSTER']}")
    print(f"Timestamp: {datetime.now()}")
    print("=" * 80)
    
    if mode == DeploymentMode.SINGLE:
        service_names = ["euglena-service"]
        task_families = ["euglena"]
        log_groups = ["/ecs/euglena"]
        repositories = ["gateway", "agent"]
    else:
        service_names = ["euglena-gateway", "euglena-agent"]
        task_families = ["euglena-gateway", "euglena-agent"]
        log_groups = ["/ecs/euglena-gateway", "/ecs/euglena-agent"]
        repositories = ["gateway", "agent"]
    
    diagnose_ecs_services(aws_config, service_names)
    diagnose_tasks(aws_config, service_names)
    diagnose_stopped_tasks(aws_config, service_names)
    diagnose_task_definitions(aws_config, task_families)
    diagnose_service_discovery(aws_config)
    diagnose_ecr_images(aws_config, repositories)
    
    if not args.skip_logs:
        diagnose_cloudwatch_logs(aws_config, log_groups, args.hours)
    
    print_section("Diagnostics Complete")
    print("\nReview the output above for issues. Common problems:")
    print("  - Tasks stopping with exit codes")
    print("  - Image pull errors")
    print("  - Health check failures")
    print("  - Network/security group issues")
    print("  - Service discovery misconfiguration")


if __name__ == "__main__":
    main()
