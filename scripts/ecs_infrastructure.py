"""
ECS infrastructure management utilities.

Can be run directly or imported as a module.
"""
import boto3
from typing import Dict, List, Optional
import sys


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
                    return True
                else:
                    print(f"  WARN: Cluster {self.cluster_name} exists but status is {status}")
                    return False
            
            print(f"  Creating cluster {self.cluster_name}...")
            self.ecs_client.create_cluster(clusterName=self.cluster_name)
            print(f"  OK: Cluster {self.cluster_name} created")
            return True
        except self.ecs_client.exceptions.ClusterNotFoundException:
            print(f"  Creating cluster {self.cluster_name}...")
            try:
                self.ecs_client.create_cluster(clusterName=self.cluster_name)
                print(f"  OK: Cluster {self.cluster_name} created")
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
        enable_az_rebalancing: bool = False
    ) -> bool:
        """
        Create ECS service if it doesn't exist, or update if it does.
        
        :param service_name: Service name.
        :param task_family: Task definition family name.
        :param network_config: Network configuration dictionary with subnets, securityGroups, assignPublicIp.
        :param desired_count: Desired task count.
        :param service_registries: Optional service discovery registries.
        :param load_balancers: Optional load balancer configuration.
        :param health_check_grace_period: Health check grace period in seconds.
        :param enable_az_rebalancing: Enable Availability Zone rebalancing.
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
                current_service = current_response.get("services", [{}])[0]
                current_desired = current_service.get("desiredCount", 0)
                
                update_params = {
                    "cluster": self.cluster_name,
                    "service": service_name,
                    "taskDefinition": task_family,
                    "desiredCount": desired_count,
                    "forceNewDeployment": True,
                    "deploymentConfiguration": deployment_config,
                    "capacityProviderStrategy": capacity_provider_strategy,
                    "platformVersion": "LATEST"
                }
                
                if health_check_grace_period is not None:
                    update_params["healthCheckGracePeriodSeconds"] = health_check_grace_period
                
                if service_registries:
                    update_params["serviceRegistries"] = service_registries
                elif current_service.get("serviceRegistries"):
                    update_params["serviceRegistries"] = current_service["serviceRegistries"]
                
                if load_balancers:
                    update_params["loadBalancers"] = load_balancers
                elif current_service.get("loadBalancers"):
                    update_params["loadBalancers"] = current_service["loadBalancers"]
                
                self.ecs_client.update_service(**update_params)
                
                if current_desired > desired_count:
                    print(f"  Scaling down from {current_desired} to {desired_count} tasks...")
                
                print(f"  OK: Service {service_name} updated")
                return True
            except Exception as e:
                print(f"  FAIL: Error updating service: {e}", file=sys.stderr)
                return False
        else:
            print(f"  Creating service {service_name}...")
            try:
                create_params = {
                    "cluster": self.cluster_name,
                    "serviceName": service_name,
                    "taskDefinition": task_family,
                    "desiredCount": desired_count,
                    "networkConfiguration": {
                        "awsvpcConfiguration": network_config
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
        enable_az_rebalancing: bool = False
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
            enable_az_rebalancing=enable_az_rebalancing
        )


def main():
    """
    Main entry point when run directly.
    """
    import argparse
    from pathlib import Path
    from dotenv import dotenv_values
    
    parser = argparse.ArgumentParser(description="Manage ECS infrastructure")
    parser.add_argument("--region", help="AWS region")
    parser.add_argument("--cluster", help="ECS cluster name")
    parser.add_argument("--services-dir", type=Path, default=None,
                       help="Services directory containing aws.env (defaults to current directory)")
    
    args = parser.parse_args()
    
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
