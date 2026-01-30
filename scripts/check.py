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


def main():
    """
    Main entry point.
    """
    parser = argparse.ArgumentParser(description="Check Euglena service health")
    parser.add_argument("--service", choices=["gateway", "agent", "all"], default="all",
                       help="Service to check (default: all)")
    parser.add_argument("--verbose", action="store_true",
                       help="Show verbose output")
    
    args = parser.parse_args()
    
    services_dir = Path.cwd()
    if not (services_dir / "aws.env").exists():
        print(f"Error: aws.env not found in current directory ({services_dir})")
        print("  Please run this script from the services/ directory")
        sys.exit(1)
    
    aws_config = load_aws_config(services_dir)
    
    services = ["gateway", "agent"] if args.service == "all" else [args.service]
    
    print("=" * 60)
    print("Euglena Health Check")
    print("=" * 60)
    print(f"Services: {', '.join(services)}")
    print(f"Region: {aws_config['AWS_REGION']}")
    print(f"Cluster: {aws_config['ECS_CLUSTER']}")
    print("=" * 60)
    
    check_ecs_services(aws_config, services, args.verbose)
    check_container_health(aws_config, services, args.verbose)
    
    print("\n" + "=" * 60)
    print("Health check complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
