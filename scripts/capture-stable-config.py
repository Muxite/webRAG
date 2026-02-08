"""
Capture current stable ECS configuration from AWS.
Creates a standardized configuration file that can be used for reference and replication.
"""
import json
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import boto3
from botocore.exceptions import ClientError

try:
    from scripts.deploy_common import load_aws_config
except ImportError:
    from deploy_common import load_aws_config


class ConfigCapturer:
    """Capture ECS configuration from AWS."""
    
    def __init__(self, region: str, cluster: str):
        self.region = region
        self.cluster = cluster
        self.ecs = boto3.client("ecs", region_name=region)
        self.elbv2 = boto3.client("elbv2", region_name=region)
        self.servicediscovery = boto3.client("servicediscovery", region_name=region)
        self.cloudwatch = boto3.client("cloudwatch", region_name=region)
        self.logs = boto3.client("logs", region_name=region)
    
    def get_cluster_config(self) -> Dict:
        """Get cluster configuration including Container Insights."""
        try:
            response = self.ecs.describe_clusters(clusters=[self.cluster])
            clusters = response.get("clusters", [])
            if not clusters:
                return {"error": "Cluster not found"}
            
            cluster = clusters[0]
            settings = cluster.get("settings", [])
            container_insights = "disabled"
            for setting in settings:
                if setting.get("name") == "containerInsights":
                    container_insights = setting.get("value", "disabled")
                    break
            
            return {
                "cluster_name": cluster.get("clusterName"),
                "cluster_arn": cluster.get("clusterArn"),
                "status": cluster.get("status"),
                "container_insights": container_insights,
                "registered_container_instances_count": cluster.get("registeredContainerInstancesCount", 0),
                "running_tasks_count": cluster.get("runningTasksCount", 0),
                "pending_tasks_count": cluster.get("pendingTasksCount", 0),
                "active_services_count": cluster.get("activeServicesCount", 0),
            }
        except ClientError as e:
            return {"error": str(e)}
    
    def get_task_definition_config(self, task_def_arn: str) -> Dict:
        """Get complete task definition configuration."""
        try:
            response = self.ecs.describe_task_definition(taskDefinition=task_def_arn)
            td = response.get("taskDefinition", {})
            
            containers = []
            for container in td.get("containerDefinitions", []):
                health_check = container.get("healthCheck", {})
                depends_on = container.get("dependsOn", [])
                
                container_config = {
                    "name": container.get("name"),
                    "image": container.get("image"),
                    "essential": container.get("essential", True),
                    "cpu": container.get("cpu"),
                    "memory": container.get("memory"),
                    "memory_reservation": container.get("memoryReservation"),
                    "port_mappings": container.get("portMappings", []),
                    "environment": container.get("environment", []),
                    "secrets": [
                        {
                            "name": s.get("name"),
                            "value_from": s.get("valueFrom"),
                        }
                        for s in container.get("secrets", [])
                    ],
                    "depends_on": [
                        {
                            "container_name": d.get("containerName"),
                            "condition": d.get("condition"),
                        }
                        for d in depends_on
                    ],
                }
                
                if health_check:
                    container_config["health_check"] = {
                        "command": health_check.get("command", []),
                        "interval": health_check.get("interval"),
                        "timeout": health_check.get("timeout"),
                        "retries": health_check.get("retries"),
                        "start_period": health_check.get("startPeriod"),
                    }
                
                containers.append(container_config)
            
            volumes = []
            for volume in td.get("volumes", []):
                vol_config = {
                    "name": volume.get("name"),
                }
                if "efsVolumeConfiguration" in volume:
                    efs = volume["efsVolumeConfiguration"]
                    vol_config["efs"] = {
                        "file_system_id": efs.get("fileSystemId"),
                        "root_directory": efs.get("rootDirectory"),
                        "transit_encryption": efs.get("transitEncryption"),
                        "authorization_config": {
                            "iam": efs.get("authorizationConfig", {}).get("iam"),
                        },
                    }
                volumes.append(vol_config)
            
            return {
                "family": td.get("family"),
                "revision": td.get("revision"),
                "cpu": td.get("cpu"),
                "memory": td.get("memory"),
                "network_mode": td.get("networkMode"),
                "task_role_arn": td.get("taskRoleArn"),
                "execution_role_arn": td.get("executionRoleArn"),
                "requires_compatibilities": td.get("requiresCompatibilities", []),
                "registered_at": td.get("registeredAt"),
                "containers": containers,
                "volumes": volumes,
            }
        except ClientError as e:
            return {"error": str(e)}
    
    def get_service_config(self, service_name: str) -> Dict:
        """Get complete service configuration."""
        try:
            response = self.ecs.describe_services(cluster=self.cluster, services=[service_name])
            services = response.get("services", [])
            if not services:
                return {"error": "Service not found"}
            
            service = services[0]
            
            # Get task definition details
            task_def_arn = service.get("taskDefinition")
            task_def_config = None
            if task_def_arn:
                task_def_config = self.get_task_definition_config(task_def_arn)
            
            # Get load balancer config
            load_balancers = []
            for lb in service.get("loadBalancers", []):
                load_balancers.append({
                    "target_group_arn": lb.get("targetGroupArn"),
                    "container_name": lb.get("containerName"),
                    "container_port": lb.get("containerPort"),
                })
            
            # Get service discovery config
            service_registries = []
            for reg in service.get("serviceRegistries", []):
                service_registries.append({
                    "registry_arn": reg.get("registryArn"),
                    "port": reg.get("port"),
                })
            
            # Get network config
            network_config = service.get("networkConfiguration", {})
            awsvpc = network_config.get("awsvpcConfiguration", {})
            
            deployment_config = service.get("deploymentConfiguration", {})
            
            return {
                "service_name": service.get("serviceName"),
                "service_arn": service.get("serviceArn"),
                "status": service.get("status"),
                "desired_count": service.get("desiredCount"),
                "running_count": service.get("runningCount"),
                "pending_count": service.get("pendingCount"),
                "task_definition": task_def_arn,
                "task_definition_config": task_def_config,
                "launch_type": service.get("launchType"),
                "platform_version": service.get("platformVersion"),
                "deployment_configuration": {
                    "maximum_percent": deployment_config.get("maximumPercent"),
                    "minimum_healthy_percent": deployment_config.get("minimumHealthyPercent"),
                    "deployment_circuit_breaker": deployment_config.get("deploymentCircuitBreaker", {}),
                },
                "health_check_grace_period_seconds": service.get("healthCheckGracePeriodSeconds"),
                "load_balancers": load_balancers,
                "service_registries": service_registries,
                "network_configuration": {
                    "awsvpc_configuration": {
                        "subnets": awsvpc.get("subnets", []),
                        "security_groups": awsvpc.get("securityGroups", []),
                        "assign_public_ip": awsvpc.get("assignPublicIp"),
                    },
                },
                "capacity_provider_strategy": service.get("capacityProviderStrategy", []),
            }
        except ClientError as e:
            return {"error": str(e)}
    
    def get_target_group_config(self, target_group_arn: str) -> Dict:
        """Get target group configuration."""
        try:
            response = self.elbv2.describe_target_groups(TargetGroupArns=[target_group_arn])
            tg = response.get("TargetGroups", [{}])[0]
            
            health_check = tg.get("HealthCheck", {})
            
            return {
                "target_group_name": tg.get("TargetGroupName"),
                "target_group_arn": tg.get("TargetGroupArn"),
                "protocol": tg.get("Protocol"),
                "port": tg.get("Port"),
                "vpc_id": tg.get("VpcId"),
                "health_check_protocol": health_check.get("Protocol"),
                "health_check_port": health_check.get("Port"),
                "health_check_path": health_check.get("Path"),
                "health_check_interval_seconds": health_check.get("IntervalSeconds"),
                "health_check_timeout_seconds": health_check.get("TimeoutSeconds"),
                "healthy_threshold_count": health_check.get("HealthyThresholdCount"),
                "unhealthy_threshold_count": health_check.get("UnhealthyThresholdCount"),
                "matcher": health_check.get("Matcher", {}),
            }
        except ClientError as e:
            return {"error": str(e)}
    
    def capture_all_config(self, service_names: List[str], target_group_arn: Optional[str] = None) -> Dict:
        """Capture complete configuration for all services."""
        config = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "region": self.region,
            "cluster": self.cluster,
            "cluster_config": self.get_cluster_config(),
            "services": {},
        }
        
        # Capture service configurations
        for service_name in service_names:
            service_config = self.get_service_config(service_name)
            if "error" not in service_config:
                config["services"][service_name] = service_config
        
        # Capture target group if provided
        if target_group_arn:
            config["target_group"] = self.get_target_group_config(target_group_arn)
        
        return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Capture stable ECS configuration from AWS"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: stable-config-DATE.json)",
    )
    parser.add_argument(
        "--services",
        type=str,
        nargs="+",
        default=["euglena-gateway", "euglena-agent"],
        help="Service names to capture (default: euglena-gateway euglena-agent)",
    )
    parser.add_argument(
        "--target-group-arn",
        type=str,
        help="Target group ARN to capture",
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
    
    print("=" * 80)
    print("CAPTURING STABLE CONFIGURATION FROM AWS")
    print("=" * 80)
    print(f"Region: {region}")
    print(f"Cluster: {cluster}")
    print(f"Services: {', '.join(args.services)}")
    if args.target_group_arn:
        print(f"Target Group: {args.target_group_arn}")
    print()
    
    # Capture configuration
    capturer = ConfigCapturer(region, cluster)
    config = capturer.capture_all_config(args.services, args.target_group_arn)
    
    # Generate output filename with date if not specified
    if args.output:
        output_path = Path(args.output)
    else:
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        output_path = Path(f"stable-config-{date_str}.json")
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    
    print(f"Configuration captured and saved to: {output_path}")
    print()
    
    # Print summary
    print("Summary:")
    cluster_config = config.get("cluster_config", {})
    if "error" not in cluster_config:
        print(f"  Cluster: {cluster_config.get('cluster_name')} ({cluster_config.get('status')})")
        print(f"  Container Insights: {cluster_config.get('container_insights')}")
    
    for service_name, service_config in config.get("services", {}).items():
        if "error" not in service_config:
            task_def = service_config.get("task_definition_config", {})
            print(f"\n  {service_name}:")
            print(f"    Status: {service_config.get('status')}")
            print(f"    Running: {service_config.get('running_count')}/{service_config.get('desired_count')}")
            if task_def:
                print(f"    Task Definition: {task_def.get('family')}:{task_def.get('revision')}")
                print(f"    CPU: {task_def.get('cpu')}, Memory: {task_def.get('memory')} MB")
                print(f"    Containers: {len(task_def.get('containers', []))}")
    
    if "target_group" in config:
        tg = config["target_group"]
        if "error" not in tg:
            print(f"\n  Target Group: {tg.get('target_group_name')}")
            print(f"    Health Check: {tg.get('health_check_path')} every {tg.get('health_check_interval_seconds')}s")
    
    print("\n" + "=" * 80)
    print(f"Full configuration saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
