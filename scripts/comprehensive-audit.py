"""
Comprehensive ECS audit and troubleshooting tool.
Combines audit-aws-changes, analyze-audit, and adds task-level investigation.
"""
import json
import sys
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
import boto3
from botocore.exceptions import ClientError

try:
    from scripts.deploy_common import load_aws_config
    from scripts.aws_audit import AWSAuditor
except ImportError:
    from deploy_common import load_aws_config
    from aws_audit import AWSAuditor


def parse_time(time_str: str) -> Optional[datetime]:
    """Parse various time string formats."""
    if not time_str:
        return None
    try:
        if "T" in time_str:
            return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S %z")
    except (ValueError, TypeError):
        return None


class TaskInvestigator:
    """Investigate specific ECS task failures."""
    
    def __init__(self, region: str, cluster: str):
        self.region = region
        self.cluster = cluster
        self.ecs = boto3.client("ecs", region_name=region)
        self.elbv2 = boto3.client("elbv2", region_name=region)
        self.cloudwatch = boto3.client("cloudwatch", region_name=region)
        self.logs = boto3.client("logs", region_name=region)
    
    def get_task_details(self, task_arn: str) -> Dict:
        """
        Get comprehensive task details including containers and stop reasons.
        
        :param task_arn: Full task ARN
        :returns: Task details dictionary
        """
        try:
            response = self.ecs.describe_tasks(cluster=self.cluster, tasks=[task_arn])
            tasks = response.get("tasks", [])
            if not tasks:
                return {"error": "Task not found"}
            
            task = tasks[0]
            containers = []
            for container in task.get("containers", []):
                containers.append({
                    "name": container.get("name"),
                    "status": container.get("lastStatus"),
                    "exit_code": container.get("exitCode"),
                    "reason": container.get("reason"),
                    "health_status": container.get("healthStatus"),
                    "cpu": container.get("cpu"),
                    "memory": container.get("memory"),
                    "memory_reservation": container.get("memoryReservation"),
                })
            
            return {
                "task_arn": task.get("taskArn"),
                "task_definition_arn": task.get("taskDefinitionArn"),
                "desired_status": task.get("desiredStatus"),
                "last_status": task.get("lastStatus"),
                "started_at": task.get("startedAt"),
                "stopped_at": task.get("stoppedAt"),
                "stopped_reason": task.get("stoppedReason"),
                "stop_code": task.get("stopCode"),
                "containers": containers,
                "cpu": task.get("cpu"),
                "memory": task.get("memory"),
                "network_interfaces": [
                    {
                        "private_ip": ni.get("privateIpv4Address"),
                        "eni_id": ni.get("attachmentId"),
                    }
                    for ni in task.get("attachments", [{}])[0].get("details", [])
                    if ni.get("name") == "networkInterfaceId"
                ],
            }
        except ClientError as e:
            return {"error": str(e)}
    
    def get_task_definition_details(self, task_def_arn: str) -> Dict:
        """
        Get task definition details including container configurations.
        
        :param task_def_arn: Task definition ARN
        :returns: Task definition details
        """
        try:
            response = self.ecs.describe_task_definition(taskDefinition=task_def_arn)
            td = response.get("taskDefinition", {})
            
            containers = []
            for container in td.get("containerDefinitions", []):
                health_check = container.get("healthCheck", {})
                containers.append({
                    "name": container.get("name"),
                    "image": container.get("image"),
                    "cpu": container.get("cpu"),
                    "memory": container.get("memory"),
                    "memory_reservation": container.get("memoryReservation"),
                    "essential": container.get("essential", True),
                    "health_check": {
                        "command": health_check.get("command", []),
                        "interval": health_check.get("interval"),
                        "timeout": health_check.get("timeout"),
                        "retries": health_check.get("retries"),
                        "start_period": health_check.get("startPeriod"),
                    } if health_check else None,
                    "depends_on": container.get("dependsOn", []),
                })
            
            return {
                "family": td.get("family"),
                "revision": td.get("revision"),
                "cpu": td.get("cpu"),
                "memory": td.get("memory"),
                "registered_at": td.get("registeredAt"),
                "containers": containers,
            }
        except ClientError as e:
            return {"error": str(e)}
    
    def get_target_group_health(self, target_group_arn: str, task_ip: Optional[str] = None) -> Dict:
        """
        Get target group health for a specific task.
        
        :param target_group_arn: Target group ARN
        :param task_ip: Task private IP (optional)
        :returns: Health check details
        """
        try:
            tg_response = self.elbv2.describe_target_groups(TargetGroupArns=[target_group_arn])
            tg = tg_response.get("TargetGroups", [{}])[0]
            
            health_response = self.elbv2.describe_target_health(TargetGroupArn=target_group_arn)
            targets = health_response.get("TargetHealthDescriptions", [])
            
            return {
                "target_group_name": tg.get("TargetGroupName"),
                "health_check": {
                    "protocol": tg.get("HealthCheckProtocol"),
                    "port": tg.get("HealthCheckPort"),
                    "path": tg.get("HealthCheckPath"),
                    "interval": tg.get("HealthCheckIntervalSeconds"),
                    "timeout": tg.get("HealthCheckTimeoutSeconds"),
                    "healthy_threshold": tg.get("HealthyThresholdCount"),
                    "unhealthy_threshold": tg.get("UnhealthyThresholdCount"),
                },
                "targets": [
                    {
                        "id": t.get("Target", {}).get("Id"),
                        "port": t.get("Target", {}).get("Port"),
                        "health": t.get("TargetHealth", {}).get("State"),
                        "reason": t.get("TargetHealth", {}).get("Reason"),
                        "description": t.get("TargetHealth", {}).get("Description"),
                    }
                    for t in targets
                ],
            }
        except ClientError as e:
            return {"error": str(e)}
    
    def get_recent_failed_tasks(self, service_name: str, hours: int = 24, max_tasks: int = 10) -> List[Dict]:
        """
        Get recent failed tasks for a service.
        
        :param service_name: ECS service name
        :param hours: Hours to look back
        :param max_tasks: Maximum tasks to return
        :returns: List of failed task details
        """
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours)
            
            response = self.ecs.list_tasks(
                cluster=self.cluster,
                serviceName=service_name,
                desiredStatus="STOPPED",
                maxResults=max_tasks,
            )
            
            failed_tasks = []
            task_arns = response.get("taskArns", [])
            
            if task_arns:
                details = self.ecs.describe_tasks(cluster=self.cluster, tasks=task_arns)
                for task in details.get("tasks", []):
                    stopped_at = task.get("stoppedAt")
                    if stopped_at:
                        if isinstance(stopped_at, str):
                            stopped_at = datetime.fromisoformat(stopped_at.replace("Z", "+00:00"))
                        if start_time <= stopped_at <= end_time:
                            stop_reason = task.get("stoppedReason", "")
                            if "failed" in stop_reason.lower() or "health" in stop_reason.lower():
                                failed_tasks.append({
                                    "task_arn": task.get("taskArn"),
                                    "stopped_at": stopped_at.isoformat(),
                                    "stopped_reason": stop_reason,
                                    "stop_code": task.get("stopCode"),
                                    "task_definition": task.get("taskDefinitionArn"),
                                    "containers": [
                                        {
                                            "name": c.get("name"),
                                            "exit_code": c.get("exitCode"),
                                            "reason": c.get("reason"),
                                        }
                                        for c in task.get("containers", [])
                                    ],
                                })
            
            return sorted(failed_tasks, key=lambda x: x["stopped_at"], reverse=True)
        except ClientError as e:
            return [{"error": str(e)}]
    
    def analyze_task_failure(self, task_arn: str, target_group_arn: Optional[str] = None) -> Dict:
        """
        Comprehensive analysis of a task failure.
        
        :param task_arn: Task ARN to analyze
        :param target_group_arn: Optional target group ARN for health check analysis
        :returns: Analysis results
        """
        task_details = self.get_task_details(task_arn)
        if "error" in task_details:
            return {"error": task_details["error"]}
        
        task_def_arn = task_details.get("task_definition_arn")
        task_def = self.get_task_definition_details(task_def_arn) if task_def_arn else {}
        
        analysis = {
            "task": task_details,
            "task_definition": task_def,
            "analysis": {
                "root_cause": [],
                "recommendations": [],
                "container_issues": [],
            },
        }
        
        # Analyze containers
        for container in task_details.get("containers", []):
            name = container.get("name")
            exit_code = container.get("exit_code")
            reason = container.get("reason", "")
            status = container.get("status")
            
            if exit_code is not None:
                if exit_code == 137:
                    analysis["analysis"]["container_issues"].append({
                        "container": name,
                        "issue": "Killed by OOM (Out of Memory)",
                        "exit_code": exit_code,
                        "recommendation": "Increase memory allocation or add memory limits",
                    })
                elif exit_code != 0:
                    analysis["analysis"]["container_issues"].append({
                        "container": name,
                        "issue": f"Exited with code {exit_code}",
                        "reason": reason,
                        "recommendation": "Check container logs for error details",
                    })
            
            if "rabbitmq" in name.lower():
                if "memory" in reason.lower() or exit_code == 137:
                    analysis["analysis"]["root_cause"].append(
                        "RabbitMQ container killed due to memory exhaustion"
                    )
                    analysis["analysis"]["recommendations"].append(
                        "Increase task memory allocation (currently may be insufficient for 5 containers)"
                    )
        
        # Analyze stop reason
        stop_reason = task_details.get("stopped_reason", "")
        if "ELB health checks" in stop_reason:
            analysis["analysis"]["root_cause"].append("ELB health check failures")
            analysis["analysis"]["recommendations"].append(
                "Check gateway container health endpoint and network connectivity"
            )
            if target_group_arn:
                tg_health = self.get_target_group_health(target_group_arn)
                analysis["target_group_health"] = tg_health
        
        # Analyze task definition
        if task_def:
            total_memory = int(task_def.get("memory", "0"))
            container_count = len(task_def.get("containers", []))
            memory_per_container = total_memory / container_count if container_count > 0 else 0
            
            analysis["analysis"]["task_resources"] = {
                "total_cpu": task_def.get("cpu"),
                "total_memory_mb": total_memory,
                "container_count": container_count,
                "memory_per_container_mb": memory_per_container,
            }
            
            if memory_per_container < 256:
                analysis["analysis"]["recommendations"].append(
                    f"Memory per container is low ({memory_per_container:.0f} MB). "
                    f"Consider increasing total memory from {total_memory} MB"
                )
            
            # Check for RabbitMQ container specifically
            for container in task_def.get("containers", []):
                if "rabbitmq" in container.get("name", "").lower():
                    rabbitmq_memory = container.get("memory_reservation") or container.get("memory") or 0
                    if rabbitmq_memory and int(rabbitmq_memory) < 256:
                        analysis["analysis"]["recommendations"].append(
                            f"RabbitMQ container memory ({rabbitmq_memory} MB) may be insufficient. "
                            f"RabbitMQ typically needs 256-512 MB minimum"
                        )
        
        return analysis


def comprehensive_audit(
    target_time: datetime,
    days: int = 10,
    task_arn: Optional[str] = None,
    target_group_arn: Optional[str] = None,
    region: str = "us-east-2",
    cluster: str = "euglena-cluster",
    account_id: str = "",
) -> Dict:
    """
    Run comprehensive audit combining all audit capabilities.
    
    :param target_time: Target time for audit
    :param days: Days to look back
    :param task_arn: Optional specific task ARN to investigate
    :param target_group_arn: Optional target group ARN
    :param region: AWS region
    :param cluster: ECS cluster name
    :param account_id: AWS account ID
    :returns: Comprehensive audit results
    """
    try:
        auditor = AWSAuditor(region, cluster, account_id)
        audit_data = auditor.audit(target_time, days)
    except Exception as e:
        print(f"Warning: Could not complete full audit: {e}")
        print("Continuing with task investigation only...")
        audit_data = {
            "target_time": target_time.isoformat(),
            "window": {"start": (target_time - timedelta(days=days)).isoformat(), "end": datetime.now(timezone.utc).isoformat()},
            "git_commits": 0,
            "cloudtrail_events": 0,
            "ecs_events": {},
            "task_definitions": {},
            "ecr_pushes": {},
            "data": {
                "commits": [],
                "cloudtrail": [],
                "ecs_events": {},
                "task_defs": {},
                "ecr_pushes": {}
            }
        }
    
    results = {
        "target_time": target_time.isoformat(),
        "window": audit_data["window"],
        "summary": audit_data,
        "task_investigation": {},
    }
    
    # Task-specific investigation
    if task_arn:
        investigator = TaskInvestigator(region, cluster)
        results["task_investigation"] = investigator.analyze_task_failure(task_arn, target_group_arn)
    
    # Analyze recent failures
    investigator = TaskInvestigator(region, cluster)
    for service in ["euglena-gateway", "euglena-agent"]:
        failed_tasks = investigator.get_recent_failed_tasks(service, hours=24)
        if failed_tasks:
            results["task_investigation"][f"{service}_recent_failures"] = failed_tasks
    
    # Analyze changes
    issues = []
    data = audit_data["data"]
    
    # Find container count changes
    for family, revs in data["task_defs"].items():
        prev_cnt = None
        for rev in sorted(revs, key=lambda x: x["time"]):
            if prev_cnt and rev["containers"] != prev_cnt:
                issues.append({
                    "type": "container_count_change",
                    "family": family,
                    "revision": rev["revision"],
                    "time": rev["time"],
                    "note": f"Containers: {prev_cnt} -> {rev['containers']}",
                })
            prev_cnt = rev["containers"]
    
    # Find resource changes
    for family, revs in data["task_defs"].items():
        prev_cpu = None
        prev_mem = None
        for rev in sorted(revs, key=lambda x: x["time"]):
            if prev_cpu and (rev["cpu"] != prev_cpu or rev["memory"] != prev_mem):
                issues.append({
                    "type": "resource_change",
                    "family": family,
                    "revision": rev["revision"],
                    "time": rev["time"],
                    "note": f"CPU: {prev_cpu}->{rev['cpu']}, Memory: {prev_mem}->{rev['memory']}",
                })
            prev_cpu = rev["cpu"]
            prev_mem = rev["memory"]
    
    results["analysis"] = {"issues": issues}
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive ECS audit and troubleshooting tool"
    )
    parser.add_argument(
        "--target-time",
        type=str,
        default=None,
        help="Target time to investigate (format: YYYY-MM-DD HH:MM:SS TZ)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=3,
        help="Number of days to look back (default: 3)",
    )
    parser.add_argument(
        "--task-arn",
        type=str,
        help="Specific task ARN to investigate",
    )
    parser.add_argument(
        "--target-group-arn",
        type=str,
        help="Target group ARN for health check analysis",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--region",
        type=str,
        help="AWS region (overrides aws.env)",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        help="ECS cluster name (overrides aws.env)",
    )
    
    args = parser.parse_args()
    
    # Load AWS config
    services_dir = Path.cwd()
    if (services_dir / "services").exists():
        services_dir = services_dir / "services"
    
    aws_config = load_aws_config(services_dir)
    region = args.region or aws_config.get("AWS_REGION", "us-east-2")
    cluster = args.cluster or aws_config.get("ECS_CLUSTER_NAME", "euglena-cluster")
    account_id = aws_config.get("AWS_ACCOUNT_ID", "")
    
    # Parse target time
    if args.target_time:
        try:
            target_time = parse_time(args.target_time)
            if not target_time:
                raise ValueError("Could not parse time")
        except ValueError:
            print(f"Error: Invalid time format: {args.target_time}", file=sys.stderr)
            sys.exit(1)
    else:
        target_time = datetime.now(timezone.utc) - timedelta(days=args.days)
    
    print("=" * 80)
    print("COMPREHENSIVE ECS AUDIT")
    print("=" * 80)
    print(f"Target time: {target_time.isoformat()}")
    print(f"Region: {region}, Cluster: {cluster}")
    print(f"Looking back: {args.days} days")
    if args.task_arn:
        print(f"Investigating task: {args.task_arn}")
    print()
    
    # Run audit
    results = comprehensive_audit(
        target_time=target_time,
        days=args.days,
        task_arn=args.task_arn,
        target_group_arn=args.target_group_arn,
        region=region,
        cluster=cluster,
        account_id=account_id,
    )
    
    # Print summary
    audit_data = results["summary"]
    print(f"Git commits: {audit_data['git_commits']}")
    print(f"CloudTrail events: {audit_data['cloudtrail_events']}")
    print(f"ECS events: {sum(audit_data['ecs_events'].values())}")
    print(f"Task definitions: {sum(audit_data['task_definitions'].values())} revisions")
    print(f"Issues found: {len(results['analysis']['issues'])}")
    
    # Print task investigation
    if results.get("task_investigation"):
        print("\n" + "=" * 80)
        print("TASK INVESTIGATION")
        print("=" * 80)
        
        if "root_cause" in results["task_investigation"].get("analysis", {}):
            print("\nRoot Causes:")
            for cause in results["task_investigation"]["analysis"]["root_cause"]:
                print(f"  - {cause}")
        
        if "container_issues" in results["task_investigation"].get("analysis", {}):
            print("\nContainer Issues:")
            for issue in results["task_investigation"]["analysis"]["container_issues"]:
                print(f"  - {issue['container']}: {issue['issue']}")
                if "recommendation" in issue:
                    print(f"    Recommendation: {issue['recommendation']}")
        
        if "recommendations" in results["task_investigation"].get("analysis", {}):
            print("\nRecommendations:")
            for rec in results["task_investigation"]["analysis"]["recommendations"]:
                print(f"  - {rec}")
        
        # Recent failures
        for key, value in results["task_investigation"].items():
            if key.endswith("_recent_failures") and value:
                service = key.replace("_recent_failures", "")
                print(f"\nRecent failures in {service}: {len(value)}")
                for task in value[:5]:
                    print(f"  - {task['stopped_at'][:19]}: {task['stopped_reason'][:60]}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\nUse --output to save detailed results to JSON file")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
