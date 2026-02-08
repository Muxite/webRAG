"""
Diagnose gateway service failures and restarts.
"""
import boto3
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dotenv import dotenv_values


def load_aws_config(services_dir: Path) -> dict:
    """Load AWS configuration."""
    aws_env_path = services_dir / "aws.env"
    if not aws_env_path.exists():
        print(f"Error: {aws_env_path} not found")
        sys.exit(1)
    return dict(dotenv_values(str(aws_env_path)))


def main():
    """Diagnose gateway failures."""
    services_dir = Path.cwd()
    if (services_dir / "services").exists():
        services_dir = services_dir / "services"
    
    aws_config = load_aws_config(services_dir)
    region = aws_config["AWS_REGION"]
    cluster = aws_config["ECS_CLUSTER"]
    service_name = "euglena-gateway"
    
    ecs = boto3.client("ecs", region_name=region)
    logs = boto3.client("logs", region_name=region)
    
    print("=== Gateway Service Status ===\n")
    
    # Check service status
    response = ecs.describe_services(cluster=cluster, services=[service_name])
    services = response.get("services", [])
    
    if not services:
        print(f"FAIL: Service {service_name} not found")
        return
    
    svc = services[0]
    desired = svc.get("desiredCount", 0)
    running = svc.get("runningCount", 0)
    pending = svc.get("pendingCount", 0)
    
    print(f"Desired: {desired}, Running: {running}, Pending: {pending}")
    
    # Recent events
    print("\n=== Recent Service Events ===")
    events = svc.get("events", [])[:10]
    for event in events:
        print(f"  {event.get('createdAt', 'N/A')}: {event.get('message', 'N/A')}")
    
    # Check stopped tasks
    print("\n=== Stopped Tasks (Last 10) ===")
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=1)
    
    response = ecs.list_tasks(
        cluster=cluster,
        serviceName=service_name,
        desiredStatus="STOPPED"
    )
    
    stopped_tasks = response.get("taskArns", [])[:10]
    if stopped_tasks:
        tasks_detail = ecs.describe_tasks(cluster=cluster, tasks=stopped_tasks)
        tasks = tasks_detail.get("tasks", [])
        
        for task in sorted(tasks, key=lambda t: t.get("stoppedAt", datetime.min.replace(tzinfo=timezone.utc)), reverse=True)[:10]:
            stopped_at = task.get("stoppedAt", "N/A")
            stop_code = task.get("stopCode", "N/A")
            stopped_reason = task.get("stoppedReason", "N/A")
            containers = task.get("containers", [])
            
            print(f"\n  Task: {task.get('taskArn', 'N/A').split('/')[-1]}")
            print(f"    Stopped: {stopped_at}")
            print(f"    Stop Code: {stop_code}")
            print(f"    Reason: {stopped_reason[:200]}")
            
            for container in containers:
                name = container.get("name", "unknown")
                exit_code = container.get("exitCode")
                reason = container.get("reason", "N/A")
                last_status = container.get("lastStatus", "N/A")
                print(f"    Container {name}: exit={exit_code}, status={last_status}, reason={reason}")
    else:
        print("  No stopped tasks found")
    
    # Check running tasks
    print("\n=== Running Tasks ===")
    response = ecs.list_tasks(cluster=cluster, serviceName=service_name, desiredStatus="RUNNING")
    running_tasks = response.get("taskArns", [])
    
    if running_tasks:
        tasks_detail = ecs.describe_tasks(cluster=cluster, tasks=running_tasks)
        tasks = tasks_detail.get("tasks", [])
        
        for task in tasks:
            started = task.get("startedAt", "N/A")
            health_status = task.get("healthStatus", "UNKNOWN")
            containers = task.get("containers", [])
            
            print(f"\n  Task: {task.get('taskArn', 'N/A').split('/')[-1]}")
            print(f"    Started: {started}")
            print(f"    Health: {health_status}")
            
            for container in containers:
                name = container.get("name", "unknown")
                health = container.get("healthStatus", "UNKNOWN")
                last_status = container.get("lastStatus", "UNKNOWN")
                print(f"    Container {name}: health={health}, status={last_status}")
    else:
        print("  No running tasks")
    
    # Check recent logs
    print("\n=== Recent Gateway Logs (Last 50 lines) ===")
    log_group = "/ecs/euglena-gateway"
    
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=10)
        
        response = logs.filter_log_events(
            logGroupName=log_group,
            startTime=int(start_time.timestamp() * 1000),
            endTime=int(end_time.timestamp() * 1000),
            limit=50
        )
        
        events = response.get("events", [])
        if events:
            for event in events[-20:]:
                msg = event.get("message", "")
                ts = datetime.fromtimestamp(event.get("timestamp", 0) / 1000)
                print(f"  {ts.strftime('%H:%M:%S')}: {msg[:150]}")
        else:
            print("  No recent logs found")
    except Exception as e:
        print(f"  Error fetching logs: {e}")
    
    # Check task definition
    print("\n=== Current Task Definition ===")
    deployments = svc.get("deployments", [])
    primary = next((d for d in deployments if d.get("status") == "PRIMARY"), None)
    if primary:
        task_def_arn = primary.get("taskDefinition", "")
        print(f"  Task Definition: {task_def_arn.split('/')[-1]}")
        
        task_def = ecs.describe_task_definition(taskDefinition=task_def_arn)
        task_def_detail = task_def.get("taskDefinition", {})
        containers = task_def_detail.get("containerDefinitions", [])
        
        print(f"  CPU: {task_def_detail.get('cpu', 'N/A')}")
        print(f"  Memory: {task_def_detail.get('memory', 'N/A')}")
        print(f"  Containers: {len(containers)}")
        
        for container in containers:
            name = container.get("name", "unknown")
            cpu = container.get("cpu", "N/A")
            memory = container.get("memory", "N/A")
            health_check = container.get("healthCheck")
            print(f"    {name}: cpu={cpu}, memory={memory}, health_check={'YES' if health_check else 'NO'}")


if __name__ == "__main__":
    main()
