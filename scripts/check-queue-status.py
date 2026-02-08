"""
Check RabbitMQ queue status and agent processing.
"""
import boto3
import sys
from pathlib import Path
from datetime import datetime, timedelta

try:
    from scripts.deploy_common import load_aws_config
except ImportError:
    from deploy_common import load_aws_config


def main():
    """Check queue and agent status."""
    services_dir = Path.cwd()
    if (services_dir / "services").exists():
        services_dir = services_dir / "services"
    
    aws_config = load_aws_config(services_dir)
    region = aws_config["AWS_REGION"]
    cluster = aws_config["ECS_CLUSTER"]
    
    ecs = boto3.client("ecs", region_name=region)
    logs = boto3.client("logs", region_name=region)
    
    print("=== Queue and Agent Status ===\n")
    
    # Check agent service
    print("1. Agent Service:")
    response = ecs.describe_services(cluster=cluster, services=["euglena-agent"])
    services = response.get("services", [])
    
    if services:
        svc = services[0]
        running = svc.get("runningCount", 0)
        desired = svc.get("desiredCount", 0)
        print(f"  Status: {running}/{desired} running")
        
        if running > 0:
            tasks = ecs.list_tasks(cluster=cluster, serviceName="euglena-agent")["taskArns"]
            if tasks:
                task = ecs.describe_tasks(cluster=cluster, tasks=[tasks[0]])["tasks"][0]
                started = task.get("startedAt")
                if started:
                    uptime = datetime.now(started.tzinfo) - started
                    print(f"  Task uptime: {uptime}")
                    print(f"  Task status: {task.get('lastStatus', 'UNKNOWN')}")
    else:
        print("  FAIL: Agent service not found")
        return
    
    # Check agent logs for recent activity
    print("\n2. Recent Agent Activity (last 5 minutes):")
    log_group = "/ecs/euglena-agent"
    
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=5)
        
        response = logs.filter_log_events(
            logGroupName=log_group,
            startTime=int(start_time.timestamp() * 1000),
            endTime=int(end_time.timestamp() * 1000),
            filterPattern="SKIP MODE",
            limit=20
        )
        
        events = response.get("events", [])
        if events:
            print(f"  Found {len(events)} skip message processing events")
            for event in events[-5:]:
                msg = event.get("message", "")[:100]
                ts = datetime.fromtimestamp(event.get("timestamp", 0) / 1000)
                print(f"    {ts.strftime('%H:%M:%S')}: {msg}")
        else:
            print("  No skip message processing found in last 5 minutes")
            print("  Checking for any agent activity...")
            
            response = logs.filter_log_events(
                logGroupName=log_group,
                startTime=int(start_time.timestamp() * 1000),
                endTime=int(end_time.timestamp() * 1000),
                limit=10
            )
            
            events = response.get("events", [])
            if events:
                print(f"  Found {len(events)} log events")
                for event in events[-3:]:
                    msg = event.get("message", "")[:80]
                    ts = datetime.fromtimestamp(event.get("timestamp", 0) / 1000)
                    print(f"    {ts.strftime('%H:%M:%S')}: {msg}")
            else:
                print("  WARN: No agent log activity found")
    except logs.exceptions.ResourceNotFoundException:
        print("  WARN: Log group not found")
    except Exception as e:
        print(f"  WARN: Could not check logs: {e}")
    
    # Check metrics service queue depth
    print("\n3. Current Queue Depth (from metrics):")
    log_group_metrics = "/ecs/euglena-gateway"
    
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=2)
        
        response = logs.filter_log_events(
            logGroupName=log_group_metrics,
            startTime=int(start_time.timestamp() * 1000),
            endTime=int(end_time.timestamp() * 1000),
            filterPattern="Queue depth",
            limit=5
        )
        
        events = response.get("events", [])
        if events:
            latest = events[-1]
            msg = latest.get("message", "")
            ts = datetime.fromtimestamp(latest.get("timestamp", 0) / 1000)
            print(f"  Latest: {ts.strftime('%H:%M:%S')} - {msg}")
        else:
            print("  No queue depth metrics found")
    except Exception as e:
        print(f"  WARN: Could not check metrics: {e}")
    
    # Check gateway logs for task submissions
    print("\n4. Recent Task Submissions (last 10 minutes):")
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=10)
        
        response = logs.filter_log_events(
            logGroupName=log_group_metrics,
            startTime=int(start_time.timestamp() * 1000),
            endTime=int(end_time.timestamp() * 1000),
            filterPattern="skipskipskip",
            limit=50
        )
        
        events = response.get("events", [])
        if events:
            print(f"  Found {len(events)} skip message submissions")
            unique_times = set()
            for event in events:
                ts = datetime.fromtimestamp(event.get("timestamp", 0) / 1000)
                unique_times.add(ts.strftime('%H:%M'))
            
            print(f"  Submissions across {len(unique_times)} minute(s)")
        else:
            print("  No skip message submissions found")
    except Exception as e:
        print(f"  WARN: Could not check gateway logs: {e}")
    
    print("\n=== Status Check Complete ===")
    print("\nInterpretation:")
    print("- Queue depth 0 = Agent is consuming messages (good)")
    print("- If no agent logs = Agent may not be running or not connected")
    print("- If no submissions = Messages may not have been queued")


if __name__ == "__main__":
    main()
