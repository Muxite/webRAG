"""
ECS service management for deployment scripts.
"""
import boto3
import time
from typing import Dict, List, Optional

try:
    from scripts.ecs_infrastructure import EcsInfrastructure
    from scripts.deployment_mode import DeploymentMode
except ImportError:
    from ecs_infrastructure import EcsInfrastructure
    from deployment_mode import DeploymentMode


def stop_old_tasks(aws_config: Dict, service_name: str) -> bool:
    """
    Stop all running tasks for a service before deployment to ensure clean state.
    
    :param aws_config: AWS configuration dictionary.
    :param service_name: Service name.
    :returns: True on success, False on error.
    """
    region = aws_config["AWS_REGION"]
    cluster = aws_config["ECS_CLUSTER"]
    ecs_client = boto3.client("ecs", region_name=region)
    
    print(f"\n=== Stopping Old Tasks for {service_name} ===")
    
    try:
        response = ecs_client.list_tasks(
            cluster=cluster,
            serviceName=service_name,
            desiredStatus="RUNNING"
        )
        task_arns = response.get("taskArns", [])
        
        if not task_arns:
            print(f"  OK: No running tasks to stop")
            return True
        
        print(f"  Found {len(task_arns)} running task(s), stopping...")
        
        for task_arn in task_arns:
            try:
                ecs_client.stop_task(
                    cluster=cluster,
                    task=task_arn,
                    reason="Stopping for clean deployment"
                )
                print(f"    Stopped task: {task_arn.split('/')[-1]}")
            except Exception as e:
                print(f"    WARN: Failed to stop task {task_arn.split('/')[-1]}: {e}")
        
        max_wait = 60
        wait_interval = 5
        waited = 0
        
        while waited < max_wait:
            response = ecs_client.list_tasks(
                cluster=cluster,
                serviceName=service_name,
                desiredStatus="RUNNING"
            )
            remaining = len(response.get("taskArns", []))
            
            if remaining == 0:
                print(f"  OK: All tasks stopped")
                return True
            
            print(f"    Waiting for {remaining} task(s) to stop... ({waited}s)")
            time.sleep(wait_interval)
            waited += wait_interval
        
        if remaining > 0:
            print(f"  WARN: {remaining} task(s) still running after {max_wait}s (continuing anyway)")
        
        return True
    except Exception as e:
        error_str = str(e)
        if "ServiceNotFoundException" in error_str or "does not exist" in error_str:
            print(f"  OK: Service does not exist yet (no tasks to stop)")
            return True
        print(f"  WARN: Error stopping old tasks: {e} (continuing anyway)")
        return True


def cleanup_old_deployments(aws_config: Dict, service_name: str) -> bool:
    """
    Clean up old non-primary deployments to ensure only the current deployment exists.
    
    :param aws_config: AWS configuration dictionary.
    :param service_name: Service name.
    :returns: True on success, False on error.
    """
    region = aws_config["AWS_REGION"]
    cluster = aws_config["ECS_CLUSTER"]
    ecs_client = boto3.client("ecs", region_name=region)
    
    print(f"\n=== Cleaning Up Old Deployments for {service_name} ===")
    
    try:
        response = ecs_client.describe_services(
            cluster=cluster,
            services=[service_name]
        )
        services = response.get("services", [])
        
        if not services:
            print(f"  OK: Service does not exist (no deployments to clean)")
            return True
        
        service = services[0]
        deployments = service.get("deployments", [])
        
        primary = next((d for d in deployments if d.get("status") == "PRIMARY"), None)
        if not primary:
            print(f"  WARN: No primary deployment found")
            return True
        
        non_primary = [d for d in deployments if d.get("id") != primary.get("id")]
        
        if not non_primary:
            print(f"  OK: No old deployments to clean (only primary exists)")
            return True
        
        print(f"  Found {len(non_primary)} old deployment(s)")
        
        for deployment in non_primary:
            dep_id = deployment.get("id", "unknown")
            status = deployment.get("status", "UNKNOWN")
            running = deployment.get("runningCount", 0)
            print(f"    Old deployment {dep_id}: status={status}, running={running}")
        
        print(f"  OK: Old deployments will be cleaned up automatically by ECS")
        return True
    except Exception as e:
        error_str = str(e)
        if "ServiceNotFoundException" in error_str:
            print(f"  OK: Service does not exist (no deployments to clean)")
            return True
        print(f"  WARN: Error checking deployments: {e} (continuing anyway)")
        return True


def ensure_exact_service_config(ecs_infrastructure: EcsInfrastructure, aws_config: Dict, 
                                service_name: str, task_family: str, network_config: Dict,
                                load_balancers: Optional[List[Dict]], desired_count: int,
                                service_registries: Optional[List[Dict]] = None) -> bool:
    """
    Ensure service has exactly the specified configuration, removing any leftover settings.
    
    :param ecs_infrastructure: EcsInfrastructure instance.
    :param aws_config: AWS configuration dictionary.
    :param service_name: Service name.
    :param task_family: Task definition family.
    :param network_config: Network configuration.
    :param load_balancers: Load balancer configuration (None to remove old config).
    :param desired_count: Desired task count.
    :param service_registries: Optional service discovery registries (None to preserve existing).
    :returns: True on success.
    """
    print(f"\n=== Ensuring Exact Service Configuration ===")
    
    cleanup_old_deployments(aws_config, service_name)
    
    if service_registries is None:
        region = aws_config["AWS_REGION"]
        cluster = aws_config["ECS_CLUSTER"]
        ecs_client = boto3.client("ecs", region_name=region)
        try:
            response = ecs_client.describe_services(cluster=cluster, services=[service_name])
            services = response.get("services", [])
            if services:
                existing_registries = services[0].get("serviceRegistries", [])
                if existing_registries:
                    service_registries = existing_registries
        except Exception:
            pass
    
    return ecs_infrastructure.ensure_service(
        service_name=service_name,
        task_family=task_family,
        network_config=network_config,
        desired_count=desired_count,
        load_balancers=load_balancers,
        service_registries=service_registries,
        health_check_grace_period=100,
        enable_az_rebalancing=True
    )


def wait_for_services_stable(aws_config: Dict, services: List[str], timeout: int = 600):
    """
    Wait for services to become stable.
    
    :param aws_config: AWS configuration dictionary.
    :param services: List of services to wait for (service names as-is).
    :param timeout: Maximum wait time in seconds.
    """
    region = aws_config["AWS_REGION"]
    cluster = aws_config["ECS_CLUSTER"]
    ecs_client = boto3.client("ecs", region_name=region)
    
    print("\n=== Waiting for Services to Stabilize ===")
    
    for service_name in services:
        print(f"\n  Waiting for {service_name}...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = ecs_client.describe_services(cluster=cluster, services=[service_name])
                services_list = response.get("services", [])
                
                if services_list:
                    svc = services_list[0]
                    deployments = svc.get("deployments", [])
                    primary = next((d for d in deployments if d.get("status") == "PRIMARY"), None)
                    
                    if primary:
                        running = primary.get("runningCount", 0)
                        desired = primary.get("desiredCount", 0)
                        pending = primary.get("pendingCount", 0)
                        
                        if running == desired and pending == 0:
                            print(f"    OK: {service_name} is stable ({running}/{desired} running)")
                            break
                        
                        print(f"    {service_name}: {running}/{desired} running, {pending} pending")
                
                time.sleep(10)
            except Exception as e:
                print(f"    Error checking service: {e}")
                time.sleep(10)
        else:
            print(f"    WARN: {service_name} did not stabilize within {timeout}s")


def stop_other_mode_services(aws_config: Dict, current_mode: DeploymentMode) -> bool:
    """
    Stop services from the other deployment mode to prevent conflicts.
    
    :param aws_config: AWS configuration dictionary.
    :param current_mode: Current deployment mode enum.
    :returns: True on success.
    """
    region = aws_config["AWS_REGION"]
    cluster = aws_config["ECS_CLUSTER"]
    ecs_client = boto3.client("ecs", region_name=region)
    
    if current_mode == DeploymentMode.SINGLE:
        other_services = ["euglena-gateway", "euglena-agent"]
        print("\n=== Stopping Autoscale Services ===")
    else:
        other_services = ["euglena-service"]
        print("\n=== Stopping Single Service ===")
    
    for service_name in other_services:
        try:
            response = ecs_client.describe_services(
                cluster=cluster,
                services=[service_name]
            )
            services = response.get("services", [])
            
            if not services:
                print(f"  OK: {service_name} does not exist (nothing to stop)")
                continue
            
            service = services[0]
            current_desired = service.get("desiredCount", 0)
            
            if current_desired == 0:
                print(f"  OK: {service_name} already stopped (desired count: 0)")
                continue
            
            print(f"  Stopping {service_name} (current desired count: {current_desired})...")
            
            ecs_client.update_service(
                cluster=cluster,
                service=service_name,
                desiredCount=0
            )
            
            print(f"  OK: Set {service_name} desired count to 0")
            
            max_wait = 120
            wait_interval = 5
            waited = 0
            
            while waited < max_wait:
                response = ecs_client.describe_services(
                    cluster=cluster,
                    services=[service_name]
                )
                services = response.get("services", [])
                if services:
                    service = services[0]
                    running = service.get("runningCount", 0)
                    if running == 0:
                        print(f"  OK: {service_name} stopped (all tasks terminated)")
                        break
                    print(f"    Waiting for {service_name} to stop... ({running} tasks running, {waited}s)")
                time.sleep(wait_interval)
                waited += wait_interval
            
            if waited >= max_wait:
                print(f"  WARN: {service_name} did not fully stop within {max_wait}s (continuing anyway)")
        
        except Exception as e:
            error_str = str(e)
            if "ServiceNotFoundException" in error_str or "does not exist" in error_str:
                print(f"  OK: {service_name} does not exist (nothing to stop)")
            else:
                print(f"  WARN: Error stopping {service_name}: {e} (continuing anyway)")
    
    return True
