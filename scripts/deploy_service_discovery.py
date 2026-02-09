"""
Service discovery setup for deployment scripts.
"""
import argparse
import boto3
from pathlib import Path
from typing import Dict, List, Optional

from deploy_common import load_aws_config


def setup_service_discovery(aws_config: Dict) -> Optional[List[Dict]]:
    """
    Set up AWS Cloud Map service discovery for gateway service.
    
    :param aws_config: AWS configuration dictionary.
    :returns: Service registry configuration or None on error.
    """
    print("\n=== Setting Up Service Discovery ===")
    
    region = aws_config["AWS_REGION"]
    cluster = aws_config["ECS_CLUSTER"]
    account_id = aws_config["AWS_ACCOUNT_ID"]
    namespace_name = aws_config.get("SERVICE_DISCOVERY_NAMESPACE", "euglena.local")
    service_name = "euglena-gateway"
    
    ecs_client = boto3.client("ecs", region_name=region)
    ec2_client = boto3.client("ec2", region_name=region)
    servicediscovery_client = boto3.client("servicediscovery", region_name=region)
    
    try:
        print("  Checking service discovery...")
        response = ecs_client.describe_services(cluster=cluster, services=[service_name])
        services = response.get("services", [])
        
        if services:
            service = services[0]
            registries = service.get("serviceRegistries", [])
            if registries:
                registry_arn = registries[0].get("registryArn", "")
                if registry_arn:
                    print(f"  OK: Service discovery already configured: {registry_arn}")
                    return registries
        
        print("  Setting up service discovery...")
        
        if not services:
            print(f"  WARN: Service {service_name} does not exist yet, will configure after creation")
            subnet_ids_str = aws_config.get("SUBNET_IDS", "")
            if not subnet_ids_str:
                print(f"  FAIL: Cannot determine subnets for service discovery")
                return None
            subnets = [s.strip() for s in subnet_ids_str.split(",") if s.strip()]
            if not subnets:
                print(f"  FAIL: No valid subnets found")
                return None
        else:
            network_config = services[0].get("networkConfiguration", {})
            awsvpc_config = network_config.get("awsvpcConfiguration", {})
            subnets = awsvpc_config.get("subnets", [])
        
        if not subnets:
            print(f"  FAIL: No subnets found in service configuration")
            return None
        
        subnet_id = subnets[0]
        response = ec2_client.describe_subnets(SubnetIds=[subnet_id])
        subnets_data = response.get("Subnets", [])
        if not subnets_data:
            print(f"  FAIL: Subnet {subnet_id} not found")
            return None
        
        vpc_id = subnets_data[0].get("VpcId")
        print(f"  VPC ID: {vpc_id}")
        
        response = servicediscovery_client.list_namespaces()
        namespaces = response.get("Namespaces", [])
        
        namespace_id = None
        for namespace in namespaces:
            if namespace.get("Name") == namespace_name:
                namespace_id = namespace.get("Id")
                print(f"  OK: Found namespace: {namespace_name} (ID: {namespace_id})")
                break
        
        if not namespace_id:
            print(f"  Creating namespace...")
            response = servicediscovery_client.create_private_dns_namespace(
                Name=namespace_name,
                Vpc=vpc_id,
                Description=f"Service discovery for {namespace_name}"
            )
            operation_id = response.get("OperationId")
            
            print("  Waiting for namespace creation...")
            waiter = servicediscovery_client.get_waiter("namespace_created")
            waiter.wait(Id=operation_id)
            
            response = servicediscovery_client.get_operation(OperationId=operation_id)
            namespace_id = response.get("Operation", {}).get("Targets", {}).get("NAMESPACE", "")
            print(f"  OK: Namespace created: {namespace_id}")
        
        response = servicediscovery_client.list_services(
            Filters=[{"Name": "NAMESPACE_ID", "Values": [namespace_id]}]
        )
        services_list = response.get("Services", [])
        
        service_id = None
        for svc in services_list:
            if svc.get("Name") == service_name:
                service_id = svc.get("Id")
                print(f"  OK: Found service: {service_name} (ID: {service_id})")
                break
        
        if not service_id:
            print("  Creating service...")
            response = servicediscovery_client.create_service(
                Name=service_name,
                NamespaceId=namespace_id,
                DnsConfig={
                    "DnsRecords": [
                        {
                            "Type": "A",
                            "TTL": 60
                        }
                    ]
                },
                HealthCheckConfig={
                    "Type": "HTTP",
                    "ResourcePath": "/health",
                    "FailureThreshold": 2
                }
            )
            service_id = response.get("Service", {}).get("Id")
            print(f"  OK: Service created: {service_id}")
        
        registry_arn = f"arn:aws:servicediscovery:{region}:{account_id}:service/{service_id}"
        
        try:
            service_response = servicediscovery_client.get_service(Id=service_id)
            service_data = service_response.get("Service", {})
            dns_config = service_data.get("DnsConfig", {})
            dns_records = dns_config.get("DnsRecords", [])
            
            if dns_records and dns_records[0].get("Type") == "SRV":
                registry_config = [{"registryArn": registry_arn}]
            else:
                registry_config = [{
                    "registryArn": registry_arn,
                    "port": 8080
                }]
        except Exception:
            registry_config = [{
                "registryArn": registry_arn,
                "port": 8080
            }]
        
        if services:
            print("  Updating ECS service with service discovery...")
            try:
                ecs_client.update_service(
                    cluster=cluster,
                    service=service_name,
                    serviceRegistries=registry_config
                )
                print(f"  OK: Service discovery configured successfully!")
            except Exception as update_error:
                error_str = str(update_error)
                if "do not require a value for 'Port'" in error_str:
                    print("  Retrying without port...")
                    registry_config = [{"registryArn": registry_arn}]
                    ecs_client.update_service(
                        cluster=cluster,
                        service=service_name,
                        serviceRegistries=registry_config
                    )
                    print(f"  OK: Service discovery configured successfully!")
                else:
                    raise
        else:
            print(f"  OK: Service discovery configured (will be applied when service is created)")
        
        return registry_config
    
    except Exception as e:
        print(f"  FAIL: Error setting up service discovery: {e}")
        import traceback
        traceback.print_exc()
        return None


def parse_args():
    """
    Parse CLI arguments.
    
    :returns: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Configure service discovery")
    parser.add_argument("--services-dir", type=Path, default=None,
                       help="Services directory containing aws.env")
    return parser.parse_args()


def main():
    """
    Main entry point.
    """
    args = parse_args()
    services_dir = args.services_dir or Path.cwd()
    aws_config = load_aws_config(services_dir)
    registries = setup_service_discovery(aws_config)
    print(registries or "None")


if __name__ == "__main__":
    main()
