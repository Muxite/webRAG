"""
Creates a comprehensive snapshot of deployment state for reproducibility.
Captures all AWS resources, configurations, and state needed to reproduce a deployment.
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from dotenv import dotenv_values

try:
    from scripts.deploy_common import load_aws_config
except ImportError:
    from deploy_common import load_aws_config


def run_aws_command(cmd: List[str], capture_json: bool = True) -> Optional[Any]:
    """Run AWS CLI command and return JSON result or None."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        if capture_json:
            return json.loads(result.stdout) if result.stdout.strip() else None
        return result.stdout.strip()
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"Warning: AWS command failed: {' '.join(cmd)} - {e}", file=sys.stderr)
        return None


def redact_secrets(data: Any, keys_to_redact: List[str] = None) -> Any:
    """Recursively redact sensitive values from data structure."""
    if keys_to_redact is None:
        keys_to_redact = ["password", "secret", "key", "token", "cookie", "arn"]
    
    if isinstance(data, dict):
        redacted = {}
        for k, v in data.items():
            key_lower = k.lower()
            if any(redact_key in key_lower for redact_key in keys_to_redact):
                redacted[k] = "***REDACTED***"
            else:
                redacted[k] = redact_secrets(v, keys_to_redact)
        return redacted
    elif isinstance(data, list):
        return [redact_secrets(item, keys_to_redact) for item in data]
    else:
        return data


def snapshot_task_definitions(region: str, families: List[str]) -> Dict:
    """Snapshot all task definitions for given families."""
    print("  Snapshotting task definitions...")
    task_defs = {}
    
    for family in families:
        # Get latest revision
        latest = run_aws_command([
            "aws", "ecs", "describe-task-definition",
            "--task-definition", family,
            "--region", region,
            "--query", "taskDefinition",
            "--no-cli-pager"
        ])
        
        if latest:
            # Also get all revisions
            revisions = run_aws_command([
                "aws", "ecs", "list-task-definitions",
                "--family-prefix", family,
                "--region", region,
                "--sort", "DESC",
                "--max-items", "10",
                "--no-cli-pager"
            ])
            
            task_defs[family] = {
                "latest": redact_secrets(latest),
                "recent_revisions": revisions.get("taskDefinitionArns", [])[:10] if revisions else []
            }
    
    return task_defs


def snapshot_ecs_services(region: str, cluster: str, services: List[str]) -> Dict:
    """Snapshot ECS service configurations."""
    print("  Snapshotting ECS services...")
    services_data = {}
    
    for service_name in services:
        service = run_aws_command([
            "aws", "ecs", "describe-services",
            "--cluster", cluster,
            "--services", service_name,
            "--region", region,
            "--no-cli-pager"
        ])
        
        if service and service.get("services"):
            svc = service["services"][0]
            services_data[service_name] = {
                "serviceArn": svc.get("serviceArn"),
                "status": svc.get("status"),
                "desiredCount": svc.get("desiredCount"),
                "runningCount": svc.get("runningCount"),
                "pendingCount": svc.get("pendingCount"),
                "taskDefinition": svc.get("taskDefinition"),
                "deploymentConfiguration": svc.get("deploymentConfiguration"),
                "networkConfiguration": svc.get("networkConfiguration"),
                "loadBalancers": svc.get("loadBalancers", []),
                "serviceRegistries": svc.get("serviceRegistries", []),
                "platformVersion": svc.get("platformVersion"),
                "launchType": svc.get("launchType"),
                "recentEvents": svc.get("events", [])[:10]
            }
            
            # Get running tasks
            tasks = run_aws_command([
                "aws", "ecs", "list-tasks",
                "--cluster", cluster,
                "--service-name", service_name,
                "--region", region,
                "--desired-status", "RUNNING",
                "--no-cli-pager"
            ])
            
            if tasks and tasks.get("taskArns"):
                task_details = run_aws_command([
                    "aws", "ecs", "describe-tasks",
                    "--cluster", cluster,
                    "--tasks", *tasks["taskArns"][:5],
                    "--region", region,
                    "--no-cli-pager"
                ])
                
                if task_details:
                    services_data[service_name]["runningTasks"] = [
                        {
                            "taskArn": t.get("taskArn"),
                            "lastStatus": t.get("lastStatus"),
                            "healthStatus": t.get("healthStatus"),
                            "containers": [
                                {
                                    "name": c.get("name"),
                                    "lastStatus": c.get("lastStatus"),
                                    "healthStatus": c.get("healthStatus"),
                                    "exitCode": c.get("exitCode"),
                                    "reason": c.get("reason")
                                }
                                for c in t.get("containers", [])
                            ],
                            "startedAt": t.get("startedAt"),
                            "cpu": t.get("cpu"),
                            "memory": t.get("memory")
                        }
                        for t in task_details.get("tasks", [])
                    ]
    
    return services_data


def snapshot_load_balancer(region: str, target_group_arn: str) -> Dict:
    """Snapshot ALB target group configuration."""
    print("  Snapshotting load balancer configuration...")
    
    # Extract target group name from ARN
    tg_name = target_group_arn.split("/")[-1] if "/" in target_group_arn else target_group_arn
    
    tg = run_aws_command([
        "aws", "elbv2", "describe-target-groups",
        "--target-group-arns", target_group_arn,
        "--region", region,
        "--no-cli-pager"
    ])
    
    tg_data = {}
    if tg and tg.get("TargetGroups"):
        tg_info = tg["TargetGroups"][0]
        tg_data = {
            "targetGroupArn": tg_info.get("TargetGroupArn"),
            "targetGroupName": tg_info.get("TargetGroupName"),
            "protocol": tg_info.get("Protocol"),
            "port": tg_info.get("Port"),
            "vpcId": tg_info.get("VpcId"),
            "healthCheckProtocol": tg_info.get("HealthCheckProtocol"),
            "healthCheckPath": tg_info.get("HealthCheckPath"),
            "healthCheckIntervalSeconds": tg_info.get("HealthCheckIntervalSeconds"),
            "healthCheckTimeoutSeconds": tg_info.get("HealthCheckTimeoutSeconds"),
            "healthyThresholdCount": tg_info.get("HealthyThresholdCount"),
            "unhealthyThresholdCount": tg_info.get("UnhealthyThresholdCount"),
            "matcher": tg_info.get("Matcher")
        }
        
        # Get target health
        health = run_aws_command([
            "aws", "elbv2", "describe-target-health",
            "--target-group-arn", target_group_arn,
            "--region", region,
            "--no-cli-pager"
        ])
        
        if health:
            tg_data["targetHealth"] = health.get("TargetHealthDescriptions", [])
    
    return tg_data


def snapshot_service_discovery(region: str, namespace_name: str) -> Dict:
    """Snapshot Cloud Map service discovery configuration."""
    print("  Snapshotting service discovery...")
    
    # Get namespace
    namespaces = run_aws_command([
        "aws", "servicediscovery", "list-namespaces",
        "--region", region,
        "--no-cli-pager"
    ])
    
    ns_data = {}
    if namespaces:
        ns = next((n for n in namespaces.get("Namespaces", []) if namespace_name in n.get("Name", "")), None)
        if ns:
            ns_id = ns.get("Id")
            ns_data["namespace"] = {
                "id": ns_id,
                "arn": ns.get("Arn"),
                "name": ns.get("Name"),
                "type": ns.get("Type")
            }
            
            # Get services in namespace
            services = run_aws_command([
                "aws", "servicediscovery", "list-services",
                "--filters", f"Name=NAMESPACE_ID,Values={ns_id}",
                "--region", region,
                "--no-cli-pager"
            ])
            
            if services:
                service_list = []
                for svc in services.get("Services", []):
                    svc_id = svc.get("Id")
                    instances = run_aws_command([
                        "aws", "servicediscovery", "list-instances",
                        "--service-id", svc_id,
                        "--region", region,
                        "--no-cli-pager"
                    ])
                    
                    service_list.append({
                        "id": svc_id,
                        "arn": svc.get("Arn"),
                        "name": svc.get("Name"),
                        "dnsConfig": svc.get("DnsConfig"),
                        "healthCheckConfig": svc.get("HealthCheckConfig"),
                        "instances": instances.get("Instances", []) if instances else []
                    })
                
                ns_data["services"] = service_list
    
    return ns_data


def snapshot_iam_roles(region: str, role_names: List[str]) -> Dict:
    """Snapshot IAM role configurations (names and ARNs only, not policies)."""
    print("  Snapshotting IAM roles...")
    roles_data = {}
    
    for role_name in role_names:
        role = run_aws_command([
            "aws", "iam", "get-role",
            "--role-name", role_name,
            "--no-cli-pager"
        ])
        
        if role and role.get("Role"):
            r = role["Role"]
            roles_data[role_name] = {
                "arn": r.get("Arn"),
                "roleName": r.get("RoleName"),
                "createDate": r.get("CreateDate"),
                "assumeRolePolicyDocument": r.get("AssumeRolePolicyDocument")
            }
            
            # Get attached policies (names only)
            policies = run_aws_command([
                "aws", "iam", "list-attached-role-policies",
                "--role-name", role_name,
                "--no-cli-pager"
            ])
            
            if policies:
                roles_data[role_name]["attachedPolicies"] = [
                    p.get("PolicyArn") for p in policies.get("AttachedPolicies", [])
                ]
    
    return roles_data


def snapshot_ecr_images(region: str, account_id: str, repositories: List[str]) -> Dict:
    """Snapshot ECR image tags and metadata."""
    print("  Snapshotting ECR images...")
    images_data = {}
    
    for repo in repositories:
        repo_name = f"euglena/{repo}" if not repo.startswith("euglena/") else repo
        images = run_aws_command([
            "aws", "ecr", "describe-images",
            "--repository-name", repo_name,
            "--region", region,
            "--max-items", "10",
            "--no-cli-pager"
        ])
        
        if images:
            images_data[repo] = [
                {
                    "imageTags": img.get("imageTags", []),
                    "imageDigest": img.get("imageDigest"),
                    "imagePushedAt": img.get("imagePushedAt"),
                    "imageSizeInBytes": img.get("imageSizeInBytes")
                }
                for img in images.get("imageDetails", [])
            ]
    
    return images_data


def snapshot_efs(region: str, file_system_ids: List[str]) -> Dict:
    """Snapshot EFS file system configurations."""
    print("  Snapshotting EFS file systems...")
    efs_data = {}
    
    for fs_id in file_system_ids:
        if not fs_id:
            continue
            
        fs = run_aws_command([
            "aws", "efs", "describe-file-systems",
            "--file-system-id", fs_id,
            "--region", region,
            "--no-cli-pager"
        ])
        
        if fs and fs.get("FileSystems"):
            fs_info = fs["FileSystems"][0]
            efs_data[fs_id] = {
                "fileSystemId": fs_info.get("FileSystemId"),
                "name": fs_info.get("Name"),
                "lifeCycleState": fs_info.get("LifeCycleState"),
                "performanceMode": fs_info.get("PerformanceMode"),
                "throughputMode": fs_info.get("ThroughputMode"),
                "encrypted": fs_info.get("Encrypted"),
                "sizeInBytes": fs_info.get("SizeInBytes")
            }
    
    return efs_data


def snapshot_network(region: str, vpc_id: str) -> Dict:
    """Snapshot VPC and network configuration."""
    print("  Snapshotting network configuration...")
    
    vpc = run_aws_command([
        "aws", "ec2", "describe-vpcs",
        "--vpc-ids", vpc_id,
        "--region", region,
        "--no-cli-pager"
    ])
    
    network_data = {}
    if vpc and vpc.get("Vpcs"):
        vpc_info = vpc["Vpcs"][0]
        network_data["vpc"] = {
            "vpcId": vpc_info.get("VpcId"),
            "cidrBlock": vpc_info.get("CidrBlock"),
            "state": vpc_info.get("State")
        }
        
        # Get subnets
        subnets = run_aws_command([
            "aws", "ec2", "describe-subnets",
            "--filters", f"Name=vpc-id,Values={vpc_id}",
            "--region", region,
            "--no-cli-pager"
        ])
        
        if subnets:
            network_data["subnets"] = [
                {
                    "subnetId": s.get("SubnetId"),
                    "cidrBlock": s.get("CidrBlock"),
                    "availabilityZone": s.get("AvailabilityZone"),
                    "mapPublicIpOnLaunch": s.get("MapPublicIpOnLaunch")
                }
                for s in subnets.get("Subnets", [])
            ]
        
        # Get security groups
        sgs = run_aws_command([
            "aws", "ec2", "describe-security-groups",
            "--filters", f"Name=vpc-id,Values={vpc_id}",
            "--region", region,
            "--no-cli-pager"
        ])
        
        if sgs:
            network_data["securityGroups"] = [
                {
                    "groupId": sg.get("GroupId"),
                    "groupName": sg.get("GroupName"),
                    "description": sg.get("Description"),
                    "ipPermissions": len(sg.get("IpPermissions", [])),
                    "ipPermissionsEgress": len(sg.get("IpPermissionsEgress", []))
                }
                for sg in sgs.get("SecurityGroups", [])
            ]
    
    return network_data


def snapshot_environment(services_dir: Path) -> Dict:
    """Snapshot environment configuration files (redacted)."""
    print("  Snapshotting environment configuration...")
    env_data = {}
    
    # Load but redact secrets
    for env_file in ["aws.env", ".env", "keys.env"]:
        env_path = services_dir / env_file
        if env_path.exists():
            env_dict = dict(dotenv_values(str(env_path)))
            env_data[env_file] = redact_secrets(env_dict)
    
    return env_data


def main():
    """Create deployment snapshot."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create deployment snapshot")
    parser.add_argument("--mode", choices=["single", "autoscale"], default="autoscale",
                       help="Deployment mode")
    parser.add_argument("--output-dir", type=str, default="deployment-snapshots",
                       help="Output directory for snapshots")
    parser.add_argument("--cluster", type=str, help="ECS cluster name (overrides aws.env)")
    parser.add_argument("--region", type=str, help="AWS region (overrides aws.env)")
    
    args = parser.parse_args()
    
    services_dir = Path.cwd()
    if (services_dir / "services").exists():
        services_dir = services_dir / "services"
    
    aws_config = load_aws_config(services_dir)
    
    region = args.region or aws_config.get("AWS_REGION", "us-east-2")
    cluster = args.cluster or aws_config.get("ECS_CLUSTER_NAME", "euglena-cluster")
    account_id = aws_config.get("AWS_ACCOUNT_ID", "")
    
    # Determine services based on mode
    if args.mode == "autoscale":
        services = ["euglena-gateway", "euglena-agent"]
        task_families = ["euglena-gateway", "euglena-agent"]
    else:
        services = ["euglena"]
        task_families = ["euglena"]
    
    # Get target group ARN from aws.env or service
    target_group_arn = aws_config.get("TARGET_GROUP_ARN", "")
    if not target_group_arn:
        # Try to get from service
        svc_data = snapshot_ecs_services(region, cluster, services[:1])
        if services[0] in svc_data and svc_data[services[0]].get("loadBalancers"):
            target_group_arn = svc_data[services[0]]["loadBalancers"][0].get("targetGroupArn", "")
    
    # Get EFS IDs
    efs_ids = []
    for key in ["EFS_FILE_SYSTEM_ID", "CHROMA_EFS_FILE_SYSTEM_ID", "RABBITMQ_EFS_FILE_SYSTEM_ID"]:
        if aws_config.get(key):
            efs_ids.append(aws_config[key])
    efs_ids = list(set(efs_ids))  # Deduplicate
    
    # Get VPC ID
    vpc_id = aws_config.get("VPC_ID", "")
    if not vpc_id:
        # Try to get from cluster
        cluster_info = run_aws_command([
            "aws", "ecs", "describe-clusters",
            "--clusters", cluster,
            "--region", region,
            "--include", "CONFIGURATIONS",
            "--no-cli-pager"
        ])
        # VPC would be in network config, but we'll skip if not available
    
    # Get IAM roles
    iam_roles = ["ecsTaskRole", "ecsTaskExecutionRole"]
    
    # Get ECR repos
    ecr_repos = ["gateway", "agent", "metrics"]
    if args.mode == "single":
        ecr_repos = ["gateway", "agent"]
    
    print(f"Creating deployment snapshot for mode: {args.mode}")
    print(f"Region: {region}, Cluster: {cluster}")
    print()
    
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "region": region,
        "cluster": cluster,
        "accountId": account_id,
        "taskDefinitions": snapshot_task_definitions(region, task_families),
        "ecsServices": snapshot_ecs_services(region, cluster, services),
        "loadBalancer": snapshot_load_balancer(region, target_group_arn) if target_group_arn else {},
        "serviceDiscovery": snapshot_service_discovery(region, "euglena.local"),
        "iamRoles": snapshot_iam_roles(region, iam_roles),
        "ecrImages": snapshot_ecr_images(region, account_id, ecr_repos),
        "efs": snapshot_efs(region, efs_ids) if efs_ids else {},
        "network": snapshot_network(region, vpc_id) if vpc_id else {},
        "environment": snapshot_environment(services_dir)
    }
    
    # Write snapshot
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"snapshot_{args.mode}_{timestamp_str}.json"
    output_path = output_dir / filename
    
    with open(output_path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    
    print(f"\nSnapshot saved to: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024:.2f} KB")
    
    # Also create a summary
    summary = {
        "timestamp": snapshot["timestamp"],
        "mode": args.mode,
        "region": region,
        "cluster": cluster,
        "services": {
            name: {
                "status": svc.get("status"),
                "desiredCount": svc.get("desiredCount"),
                "runningCount": svc.get("runningCount"),
                "taskDefinition": svc.get("taskDefinition")
            }
            for name, svc in snapshot["ecsServices"].items()
        },
        "taskDefinitions": {
            family: {
                "latestRevision": td["latest"].get("revision") if td.get("latest") else None,
                "recentRevisions": len(td.get("recent_revisions", []))
            }
            for family, td in snapshot["taskDefinitions"].items()
        },
        "loadBalancer": {
            "targetGroupName": snapshot["loadBalancer"].get("targetGroupName"),
            "healthyTargets": len([t for t in snapshot["loadBalancer"].get("targetHealth", []) if t.get("TargetHealth", {}).get("State") == "healthy"])
        } if snapshot.get("loadBalancer") else {},
        "serviceDiscovery": {
            "namespace": snapshot["serviceDiscovery"].get("namespace", {}).get("name"),
            "services": len(snapshot["serviceDiscovery"].get("services", []))
        } if snapshot.get("serviceDiscovery") else {}
    }
    
    summary_path = output_dir / f"summary_{args.mode}_{timestamp_str}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
