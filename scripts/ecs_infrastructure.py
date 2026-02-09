"""
ECS infrastructure management utilities.

Can be run directly or imported as a module.
"""
import boto3
from typing import Dict, List, Optional
import sys
import time


class EcsInfrastructure:
    """
    Manages ECS clusters and services (create or update).
    
    :param region: AWS region name.
    :param cluster_name: ECS cluster name.
    """
    
    def __init__(self, region: str, cluster_name: str):
        self.region = region
        self.cluster_name = cluster_name
        self.ecs_client = boto3.client("ecs", region_name=region)
    
    def ensure_cluster(self) -> bool:
        """
        Ensure ECS cluster exists, create if it doesn't.
        Enables Container Insights for task-level metrics.
        
        :returns: True on success, False on error.
        """
        try:
            response = self.ecs_client.describe_clusters(clusters=[self.cluster_name])
            clusters = response.get("clusters", [])
            
            if clusters:
                cluster = clusters[0]
                status = cluster.get("status", "UNKNOWN")
                if status == "ACTIVE":
                    print(f"  OK: Cluster {self.cluster_name} exists and is active")
                    settings = cluster.get("settings", [])
                    container_insights_enabled = any(
                        s.get("name") == "containerInsights" and s.get("value") == "enabled"
                        for s in settings
                    )
                    if not container_insights_enabled:
                        print(f"  Enabling Container Insights on cluster {self.cluster_name}...")
                        try:
                            self.ecs_client.update_cluster_settings(
                                cluster=self.cluster_name,
                                settings=[{"name": "containerInsights", "value": "enabled"}]
                            )
                            print(f"  OK: Container Insights enabled")
                        except Exception as e:
                            print(f"  WARN: Failed to enable Container Insights: {e}")
                    else:
                        print(f"  OK: Container Insights already enabled")
                    return True
                else:
                    print(f"  WARN: Cluster {self.cluster_name} exists but status is {status}")
                    return False
            
            print(f"  Creating cluster {self.cluster_name} with Container Insights...")
            self.ecs_client.create_cluster(
                clusterName=self.cluster_name,
                settings=[{"name": "containerInsights", "value": "enabled"}]
            )
            print(f"  OK: Cluster {self.cluster_name} created with Container Insights")
            return True
        except self.ecs_client.exceptions.ClusterNotFoundException:
            print(f"  Creating cluster {self.cluster_name} with Container Insights...")
            try:
                self.ecs_client.create_cluster(
                    clusterName=self.cluster_name,
                    settings=[{"name": "containerInsights", "value": "enabled"}]
                )
                print(f"  OK: Cluster {self.cluster_name} created with Container Insights")
                return True
            except Exception as e:
                print(f"  FAIL: Error creating cluster: {e}", file=sys.stderr)
                return False
        except Exception as e:
            print(f"  FAIL: Error checking cluster: {e}", file=sys.stderr)
            return False
    
    def service_exists(self, service_name: str) -> bool:
        """
        Check if ECS service exists.
        
        :param service_name: Service name to check.
        :returns: True if service exists, False otherwise.
        """
        try:
            response = self.ecs_client.describe_services(
                cluster=self.cluster_name,
                services=[service_name]
            )
            services = response.get("services", [])
            return len(services) > 0
        except Exception:
            return False
    
    def create_or_update_service(
        self,
        service_name: str,
        task_family: str,
        network_config: Dict,
        desired_count: int = 1,
        service_registries: Optional[List[Dict]] = None,
        load_balancers: Optional[List[Dict]] = None,
        health_check_grace_period: Optional[int] = None,
        enable_az_rebalancing: bool = False,
        force_deploy: bool = False
    ) -> bool:
        """
        Create ECS service if it doesn't exist, or update if it does.
        
        Handles service lifecycle: checks if service exists, waits for ACTIVE state if updating,
        creates new service if it doesn't exist, and retries on ServiceNotActiveException.
        
        :param service_name: Service name.
        :param task_family: Task definition family name.
        :param network_config: Network configuration dictionary with subnets, securityGroups, assignPublicIp.
        :param desired_count: Desired task count.
        :param service_registries: Optional service discovery registries.
        :param load_balancers: Optional load balancer configuration.
        :param health_check_grace_period: Health check grace period in seconds.
        :param enable_az_rebalancing: Enable Availability Zone rebalancing.
        :param force_deploy: Force a new deployment even when task definition is unchanged.
        :returns: True on success, False on error.
        """
        exists = self.service_exists(service_name)
        
        deployment_config = {
            "deploymentCircuitBreaker": {
                "enable": True,
                "rollback": True
            }
        }
        
        if enable_az_rebalancing:
            deployment_config["alarms"] = {
                "alarmNames": [],
                "enable": False,
                "rollback": False
            }
        
        capacity_provider_strategy = [
            {
                "capacityProvider": "FARGATE",
                "base": 0,
                "weight": 1
            }
        ]
        
        if exists:
            print(f"  Updating service {service_name}...")
            try:
                current_response = self.ecs_client.describe_services(
                    cluster=self.cluster_name,
                    services=[service_name]
                )
                services = current_response.get("services", [])
                if not services:
                    exists = False
                else:
                    current_service = services[0]
                    service_status = current_service.get("status", "UNKNOWN")
                    current_desired = current_service.get("desiredCount", 0)
                    
                    if service_status == "INACTIVE":
                        print(f"  Service is INACTIVE, will create new service")
                        exists = False
                    elif service_status != "ACTIVE":
                        print(f"  Service status: {service_status}, waiting for ACTIVE state...")
                        max_wait = 300
                        wait_interval = 10
                        waited = 0
                        
                        while service_status != "ACTIVE" and waited < max_wait:
                            time.sleep(wait_interval)
                            waited += wait_interval
                            
                            current_response = self.ecs_client.describe_services(
                                cluster=self.cluster_name,
                                services=[service_name]
                            )
                            services = current_response.get("services", [])
                            if services:
                                current_service = services[0]
                                service_status = current_service.get("status", "UNKNOWN")
                                if service_status == "INACTIVE":
                                    print(f"    Service became INACTIVE, will create new service")
                                    exists = False
                                    break
                                print(f"    Status: {service_status} (waited {waited}s)")
                            else:
                                exists = False
                                break
                        
                        if service_status != "ACTIVE" and waited >= max_wait and exists:
                            print(f"  WARN: Service did not become ACTIVE within {max_wait}s (status: {service_status})")
                            print(f"  Attempting update anyway...")
                    
                    if exists:
                        current_task_def = current_service.get("taskDefinition", "")
                        latest_task_def_arn = None
                        try:
                            latest_response = self.ecs_client.describe_task_definition(taskDefinition=task_family)
                            latest_task_def_arn = latest_response.get("taskDefinition", {}).get("taskDefinitionArn", "")
                        except Exception:
                            pass
                        
                        task_def_changed = current_task_def != latest_task_def_arn if latest_task_def_arn else True
                        should_force = force_deploy or task_def_changed
                        
                        awsvpc_config = {
                            "subnets": network_config.get("subnets", []),
                            "securityGroups": network_config.get("securityGroups", []),
                            "assignPublicIp": network_config.get("assignPublicIp", "ENABLED")
                        }
                        
                        update_params = {
                            "cluster": self.cluster_name,
                            "service": service_name,
                            "taskDefinition": task_family,
                            "desiredCount": desired_count,
                            "forceNewDeployment": should_force,
                            "deploymentConfiguration": deployment_config,
                            "capacityProviderStrategy": capacity_provider_strategy,
                            "platformVersion": "LATEST",
                            "networkConfiguration": {
                                "awsvpcConfiguration": awsvpc_config
                            }
                        }
                        
                        if force_deploy:
                            print("  Forcing new deployment (task definition may be unchanged)")
                        elif not task_def_changed:
                            print("  Task definition unchanged, skipping force deployment")
                        
                        if health_check_grace_period is not None:
                            update_params["healthCheckGracePeriodSeconds"] = health_check_grace_period
                        
                        if service_registries is not None:
                            update_params["serviceRegistries"] = service_registries
                        elif current_service.get("serviceRegistries"):
                            update_params["serviceRegistries"] = []
                        
                        if load_balancers is not None:
                            update_params["loadBalancers"] = load_balancers
                        elif current_service.get("loadBalancers"):
                            update_params["loadBalancers"] = []
                        
                        try:
                            self.ecs_client.update_service(**update_params)
                            
                            if current_desired > desired_count:
                                print(f"  Scaling down from {current_desired} to {desired_count} tasks...")
                            
                            print(f"  OK: Service {service_name} updated")
                            return True
                        except Exception as update_error:
                            error_str = str(update_error)
                            if "ServiceNotActiveException" in error_str or "Service was not ACTIVE" in error_str:
                                print(f"  WARN: Service not in ACTIVE state, waiting and retrying...")
                                time.sleep(30)
                                try:
                                    retry_response = self.ecs_client.describe_services(
                                        cluster=self.cluster_name,
                                        services=[service_name]
                                    )
                                    retry_services = retry_response.get("services", [])
                                    if retry_services and retry_services[0].get("status") == "ACTIVE":
                                        self.ecs_client.update_service(**update_params)
                                        print(f"  OK: Service {service_name} updated (after retry)")
                                        return True
                                except:
                                    pass
                                print(f"  FAIL: Service {service_name} is not in ACTIVE state and could not be updated")
                                return False
                            else:
                                raise
            except Exception as e:
                error_str = str(e)
                if "ServiceNotActiveException" in error_str or "Service was not ACTIVE" in error_str:
                    print(f"  WARN: Service not in ACTIVE state, waiting and retrying...")
                    time.sleep(30)
                    try:
                        retry_response = self.ecs_client.describe_services(
                            cluster=self.cluster_name,
                            services=[service_name]
                        )
                        retry_services = retry_response.get("services", [])
                        if retry_services and retry_services[0].get("status") == "ACTIVE":
                            return self.ensure_service(
                                service_name, task_family, network_config,
                                desired_count, load_balancers, health_check_grace_period,
                                service_registries, enable_az_rebalancing
                            )
                    except:
                        pass
                    print(f"  FAIL: Service {service_name} is not in ACTIVE state and could not be updated")
                    return False
                else:
                    print(f"  FAIL: Error updating service: {e}", file=sys.stderr)
                return False
        
        if not exists:
            print(f"  Creating service {service_name}...")
            try:
                awsvpc_config = {
                    "subnets": network_config.get("subnets", []),
                    "securityGroups": network_config.get("securityGroups", []),
                    "assignPublicIp": network_config.get("assignPublicIp", "ENABLED")
                }
                
                create_params = {
                    "cluster": self.cluster_name,
                    "serviceName": service_name,
                    "taskDefinition": task_family,
                    "desiredCount": desired_count,
                    "networkConfiguration": {
                        "awsvpcConfiguration": awsvpc_config
                    },
                    "deploymentConfiguration": deployment_config,
                    "capacityProviderStrategy": capacity_provider_strategy,
                    "platformVersion": "LATEST"
                }
                
                if health_check_grace_period is not None:
                    create_params["healthCheckGracePeriodSeconds"] = health_check_grace_period
                
                if service_registries:
                    create_params["serviceRegistries"] = service_registries
                
                if load_balancers:
                    create_params["loadBalancers"] = load_balancers
                
                self.ecs_client.create_service(**create_params)
                print(f"  OK: Service {service_name} created")
                return True
            except Exception as e:
                print(f"  FAIL: Error creating service: {e}", file=sys.stderr)
                return False
    
    def ensure_service(
        self,
        service_name: str,
        task_family: str,
        network_config: Dict,
        desired_count: int = 1,
        service_registries: Optional[List[Dict]] = None,
        load_balancers: Optional[List[Dict]] = None,
        health_check_grace_period: Optional[int] = None,
        enable_az_rebalancing: bool = False,
        force_deploy: bool = False
    ) -> bool:
        """
        Ensure ECS service exists with correct configuration.
        Alias for create_or_update_service for clarity.
        
        :param service_name: Service name.
        :param task_family: Task definition family name.
        :param network_config: Network configuration dictionary.
        :param desired_count: Desired task count.
        :param service_registries: Optional service discovery registries.
        :param load_balancers: Optional load balancer configuration.
        :param health_check_grace_period: Health check grace period in seconds.
        :param enable_az_rebalancing: Enable Availability Zone rebalancing.
        :param force_deploy: Force a new deployment even when task definition is unchanged.
        :returns: True on success, False on error.
        """
        return self.create_or_update_service(
            service_name=service_name,
            task_family=task_family,
            network_config=network_config,
            desired_count=desired_count,
            service_registries=service_registries,
            load_balancers=load_balancers,
            health_check_grace_period=health_check_grace_period,
            enable_az_rebalancing=enable_az_rebalancing,
            force_deploy=force_deploy
        )


def parse_args():
    """
    Parse CLI arguments.

    :returns: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Manage ECS infrastructure")
    parser.add_argument("--region", help="AWS region")
    parser.add_argument("--cluster", help="ECS cluster name")
    parser.add_argument("--services-dir", type=Path, default=None,
                       help="Services directory containing aws.env (defaults to current directory)")
    
    return parser.parse_args()

def main():
    """
    Main entry point when run directly.
    """
    import argparse
    from pathlib import Path
    from dotenv import dotenv_values
    
    args = parse_args()
    
    services_dir = args.services_dir or Path.cwd()
    aws_env_path = services_dir / "aws.env"
    
    if aws_env_path.exists():
        aws_config = dict(dotenv_values(str(aws_env_path)))
        region = args.region or aws_config.get("AWS_REGION")
        cluster = args.cluster or aws_config.get("ECS_CLUSTER")
    else:
        if not args.region or not args.cluster:
            print("Error: --region and --cluster required or aws.env must exist", file=sys.stderr)
            sys.exit(1)
        region = args.region
        cluster = args.cluster
    
    if not region or not cluster:
        print("Error: Region and cluster name required", file=sys.stderr)
        sys.exit(1)
    
    infrastructure = EcsInfrastructure(region=region, cluster_name=cluster)
    
    print(f"=== ECS Infrastructure ===")
    print(f"Region: {region}")
    print(f"Cluster: {cluster}")
    
    if infrastructure.ensure_cluster():
        print("\nOK: Cluster ready")
    else:
        print("\nFAIL: Cluster setup failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
