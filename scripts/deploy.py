"""
Deployment script for Euglena services.
"""
import boto3
import json
import subprocess
import sys
import time
import argparse
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
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


def run_command(cmd: List[str], check: bool = True, capture: bool = False, shell: bool = False) -> Tuple[bool, str, str]:
    """
    Run a shell command and return result.
    
    :param cmd: Command as list of strings or string.
    :param check: Whether to raise on non-zero exit.
    :param capture: Whether to capture output.
    :param shell: Whether to run in shell.
    :returns: Tuple of (success, stdout, stderr).
    """
    if isinstance(cmd, str) and not shell:
        cmd = cmd.split()
    
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture,
            text=True,
            shell=shell
        )
        return True, result.stdout if capture else "", result.stderr if capture else ""
    except subprocess.CalledProcessError as e:
        return False, e.stdout if capture else "", e.stderr if capture else str(e)
    except FileNotFoundError:
        return False, "", f"Command not found: {cmd[0] if isinstance(cmd, list) else cmd}"


def push_to_ecr(services_dir: Path, aws_config: Dict, services: List[str]) -> bool:
    """
    Push Docker images to ECR.
    
    :param services_dir: Services directory path.
    :param aws_config: AWS configuration dictionary.
    :param services: List of services to push (gateway, agent).
    :returns: True on success.
    """
    account_id = aws_config["AWS_ACCOUNT_ID"]
    region = aws_config["AWS_REGION"]
    
    print("\n=== Pushing Images to ECR ===")
    
    registry_url = f"{account_id}.dkr.ecr.{region}.amazonaws.com"
    
    print("Authenticating with ECR...")
    cmd = f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {registry_url}"
    success, _, stderr = run_command(cmd, check=False, shell=True)
    if not success:
        print(f"  FAIL: ECR login failed: {stderr}")
        return False
    print("  OK: ECR login successful")
    
    for service in services:
        print(f"Processing {service}...")
        
        repository_name = f"euglena/{service}"
        
        print("  Checking repository...")
        check_cmd = ["aws", "ecr", "describe-repositories", "--repository-names", repository_name, "--region", region]
        success, _, _ = run_command(check_cmd, check=False, capture=True)
        if not success:
            create_cmd = ["aws", "ecr", "create-repository", "--repository-name", repository_name, "--region", region, "--image-scanning-configuration", "scanOnPush=true"]
            success, _, stderr = run_command(create_cmd, check=False, capture=True)
            if not success:
                print(f"    FAIL: Failed to create repository: {stderr}")
                continue
            print(f"    OK: Created repository")
        else:
            print(f"    OK: Repository exists")
        
        dockerfile_path = services_dir / service / ".dockerfile"
        if not dockerfile_path.exists():
            print(f"    FAIL: Dockerfile not found: {dockerfile_path}")
            continue
        
        print("  Building image...")
        image_name = f"euglena/{service}"
        dockerfile_relative = f"{service}/.dockerfile"
        build_cmd = ["docker", "build", "-f", dockerfile_relative, "-t", image_name, "."]
        
        original_dir = os.getcwd()
        try:
            os.chdir(services_dir)
            success, _, stderr = run_command(build_cmd, check=False, capture=True)
            if not success:
                print(f"    FAIL: Build failed: {stderr}")
                continue
        finally:
            os.chdir(original_dir)
        print(f"    OK: Image built")
        
        print(f"  Tagging and pushing...")
        ecr_image = f"{registry_url}/euglena/{service}:latest"
        success, _, stderr = run_command(["docker", "tag", image_name, ecr_image], check=False, capture=True)
        if not success:
            print(f"    FAIL: Tag failed: {stderr}")
            continue
        
        success, _, stderr = run_command(["docker", "push", ecr_image], check=False, capture=True)
        if not success:
            print(f"    FAIL: Push failed: {stderr}")
            continue
        print(f"    OK: Image pushed to ECR")
    
    return True


def package_lambda(services_dir: Path) -> bool:
    """
    Package Lambda function for deployment.
    
    :param services_dir: Services directory path.
    :returns: True on success.
    """
    print("\n=== Packaging Lambda Function ===")
    
    project_root = services_dir.parent
    lambda_dir = services_dir / "lambda_autoscaling"
    output_dir = project_root / "dist"
    
    if not lambda_dir.exists():
        print(f"  WARN: Lambda directory not found: {lambda_dir}")
        print("  Skipping Lambda packaging")
        return True
    
    output_dir.mkdir(exist_ok=True)
    package_name = f"lambda-autoscaling-{datetime.now().strftime('%Y%m%d%H%M%S')}.zip"
    package_path = output_dir / package_name
    
    temp_dir = output_dir / "lambda-package"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    
    print("  Copying files...")
    shutil.copy(lambda_dir / "lambda_function.py", temp_dir / "lambda_function.py")
    shutil.copy(services_dir / "aws.env", temp_dir / "aws.env")
    if (services_dir / ".env").exists():
        shutil.copy(services_dir / ".env", temp_dir / ".env")
    
    shared_dir = services_dir / "shared"
    if shared_dir.exists():
        shared_dest = temp_dir / "shared"
        shutil.copytree(shared_dir, shared_dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests"))
    
    requirements = lambda_dir / "requirements.txt"
    if requirements.exists():
        print("  Installing dependencies...")
        success, _, stderr = run_command([
            sys.executable, "-m", "pip", "install",
            "-r", str(requirements),
            "-t", str(temp_dir),
            "--quiet"
        ], check=False, capture=True)
        if not success:
            print(f"    WARN: Dependency installation had issues: {stderr}")
    
    print("  Creating ZIP package...")
    with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(temp_dir)
                zipf.write(file_path, arcname)
    
    shutil.rmtree(temp_dir)
    
    size_mb = package_path.stat().st_size / (1024 * 1024)
    print(f"  OK: Package created: {package_path.name} ({size_mb:.2f} MB)")
    print(f"\n  NOTE: Lambda function must be deployed manually.")
    print(f"  Upload {package_path.name} to AWS Lambda via Console or CLI:")
    print(f"  aws lambda update-function-code --function-name <function-name> --zip-file fileb://{package_path}")
    
    return True


def build_and_register_task_definitions(services_dir: Path) -> bool:
    """
    Build and register task definitions.
    
    :param services_dir: Services directory path.
    :returns: True on success.
    """
    print("\n=== Building Task Definitions ===")
    
    script_path = services_dir.parent / "scripts" / "build-task-definition.py"
    success, stdout, stderr = run_command(
        ["python", str(script_path)],
        check=False,
        capture=True
    )
    
    if not success:
        print(f"  FAIL: Failed to build task definitions: {stderr}")
        return False
    
    print("  OK: Task definitions built and registered")
    return True


def update_services(services_dir: Path, aws_config: Dict, services: List[str]) -> bool:
    """
    Update ECS services to use latest task definitions.
    
    :param services_dir: Services directory path.
    :param aws_config: AWS configuration dictionary.
    :param services: List of services to update.
    :returns: True on success.
    """
    print("\n=== Updating ECS Services ===")
    
    region = aws_config["AWS_REGION"]
    cluster = aws_config["ECS_CLUSTER"]
    ecs_client = boto3.client("ecs", region_name=region)
    
    for service in services:
        service_name = f"euglena-{service}"
        task_family = f"euglena-{service}"
        
        print(f"\n  Updating {service_name}...")
        
        try:
            current_response = ecs_client.describe_services(cluster=cluster, services=[service_name])
            current_svc = current_response.get("services", [{}])[0]
            current_task_def = current_svc.get("taskDefinition", "")
            service_registries = current_svc.get("serviceRegistries", [])
            
            latest_response = ecs_client.describe_task_definition(taskDefinition=task_family)
            latest_task_def = latest_response.get("taskDefinition", {}).get("taskDefinitionArn", "")
            
            if current_task_def == latest_task_def:
                print(f"    WARN: Service already using latest task definition")
                print(f" Forcing new deployment anyway (forceNewDeployment=True)")
            else:
                print(f"    Updating from {current_task_def.split('/')[-1]} to {latest_task_def.split('/')[-1]}")
        except Exception as e:
            print(f"    WARN: Could not check current task definition: {e}")
            service_registries = []
        
        deployment_config = {
            "deploymentCircuitBreaker": {
                "enable": True,
                "rollback": True
            }
        }
        
        update_params = {
            "cluster": cluster,
            "service": service_name,
            "taskDefinition": task_family,
            "forceNewDeployment": True,
            "deploymentConfiguration": deployment_config
        }
        
        if service_registries:
            update_params["serviceRegistries"] = service_registries
        
        try:
            ecs_client.update_service(**update_params)
            print(f"    OK: {service_name} rolling deployment initiated")
        except Exception as e:
            print(f"    FAIL: Failed to update {service_name}: {e}")
            return False
    
    return True


def setup_service_discovery(aws_config: Dict) -> bool:
    """
    Set up AWS Cloud Map service discovery for gateway service.
    
    :param aws_config: AWS configuration dictionary.
    :returns: True on success.
    """
    print("\n=== Setting Up Service Discovery ===")
    
    region = aws_config["AWS_REGION"]
    cluster = aws_config["ECS_CLUSTER"]
    account_id = aws_config["AWS_ACCOUNT_ID"]
    namespace_name = aws_config.get("SERVICE_DISCOVERY_NAMESPACE", "euglena.local")
    service_name = f"euglena-gateway"
    
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
                    return True
        
        print("  Setting up service discovery...")
        
        network_config = services[0].get("networkConfiguration", {}) if services else {}
        awsvpc_config = network_config.get("awsvpcConfiguration", {})
        subnets = awsvpc_config.get("subnets", [])
        
        if not subnets:
            print(f"  FAIL: No subnets found in service configuration")
            return False
        
        subnet_id = subnets[0]
        response = ec2_client.describe_subnets(SubnetIds=[subnet_id])
        subnets_data = response.get("Subnets", [])
        if not subnets_data:
            print(f"  FAIL: Subnet {subnet_id} not found")
            return False
        
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
        
        response = servicediscovery_client.list_services(NamespaceId=namespace_id)
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
        
        print("  Updating ECS service...")
        ecs_client.update_service(
            cluster=cluster,
            service=service_name,
            serviceRegistries=[{
                "registryArn": registry_arn,
                "port": 8080
            }]
        )
        print(f"  OK: Service discovery configured successfully!")
        return True
    
    except Exception as e:
        print(f"  FAIL: Error setting up service discovery: {e}")
        import traceback
        traceback.print_exc()
        return False


def setup_autoscaling_rule(aws_config: Dict) -> bool:
    """
    Set up EventBridge rule for autoscaling Lambda.
    
    :param aws_config: AWS configuration dictionary.
    :returns: True on success.
    """
    print("\n=== Setting Up Autoscaling Rule ===")
    
    region = aws_config["AWS_REGION"]
    rule_name = aws_config.get("EVENTBRIDGE_RULE_NAME", "euglena-autoscaling-trigger")
    lambda_name = aws_config.get("LAMBDA_FUNCTION_NAME", "euglena-autoscaling")
    schedule = aws_config.get("AUTOSCALING_SCHEDULE", "1 minute")
    
    eventbridge = boto3.client("events", region_name=region)
    lambda_client = boto3.client("lambda", region_name=region)
    
    try:
        schedule_expr = f"rate({schedule})"
        
        
        try:
            rule = eventbridge.describe_rule(Name=rule_name)
            print(f"  Found existing rule: {rule_name}")
        except eventbridge.exceptions.ResourceNotFoundException:
            print(f"  Creating new rule: {rule_name}")
        
        print("  Updating rule...")
        eventbridge.put_rule(
            Name=rule_name,
            ScheduleExpression=schedule_expr,
            State="ENABLED",
            Description="Triggers euglena autoscaling Lambda function"
        )
        print(f"  OK: Rule updated")
        
        try:
            lambda_arn = lambda_client.get_function(FunctionName=lambda_name)["Configuration"]["FunctionArn"]
            target_id = "autoscaling-lambda-target"
            
            targets = eventbridge.list_targets_by_rule(Rule=rule_name)
            existing_targets = {t["Id"]: t for t in targets.get("Targets", [])}
            
            if target_id not in existing_targets:
                print("  Adding Lambda target...")
                eventbridge.put_targets(
                    Rule=rule_name,
                    Targets=[{
                        "Id": target_id,
                        "Arn": lambda_arn
                    }]
                )
                print(f"  OK: Target added")
            else:
                existing_arn = existing_targets[target_id]["Arn"]
                if existing_arn != lambda_arn:
                    print("  Updating Lambda target...")
                    eventbridge.remove_targets(Rule=rule_name, Ids=[target_id])
                    eventbridge.put_targets(
                        Rule=rule_name,
                        Targets=[{
                            "Id": target_id,
                            "Arn": lambda_arn
                        }]
                    )
                    print(f"  OK: Target updated")
                else:
                    print(f"  OK: Target already configured correctly")
        except lambda_client.exceptions.ResourceNotFoundException:
            print(f"  WARN: Lambda '{lambda_name}' not found. Rule updated but target not configured.")
            return False
        
        rule = eventbridge.describe_rule(Name=rule_name)
        if rule.get("State") == "ENABLED":
            print(f"  OK: Autoscaling rule is ENABLED")
            return True
        else:
            print(f"  FAIL: Rule state is not ENABLED: {rule.get('State')}")
            return False
    
    except Exception as e:
        print(f"  FAIL: Error setting up autoscaling rule: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_and_fix_network(aws_config: Dict) -> bool:
    """
    Validate network configuration and fix if needed.
    
    :param aws_config: AWS configuration dictionary.
    :returns: True if network is valid.
    """
    print("\n=== Validating Network Configuration ===")
    
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from network_utils import validate_network_configuration, fix_security_group_rules
    
    is_valid, issues = validate_network_configuration(aws_config)
    
    if is_valid:
        print("  OK: Network configuration is valid")
        return True
    
    print(f"  WARN: Found {len(issues)} network issues")
    print("  Attempting to fix security group rules...")
    
    success, messages = fix_security_group_rules(aws_config)
    if success:
        print("  OK: Security group rules fixed")
        time.sleep(5)
        is_valid, issues = validate_network_configuration(aws_config)
        if is_valid:
            print("  OK: Network configuration is now valid")
            return True
        else:
            print(f"  WARN: Network still has {len(issues)} issues after fix")
            return False
    else:
        print(f"  FAIL: Failed to fix: {messages}")
        return False


def wait_for_services_stable(aws_config: Dict, services: List[str], timeout: int = 600):
    """
    Wait for services to become stable.
    
    :param aws_config: AWS configuration dictionary.
    :param services: List of services to wait for.
    :param timeout: Maximum wait time in seconds.
    """
    region = aws_config["AWS_REGION"]
    cluster = aws_config["ECS_CLUSTER"]
    ecs_client = boto3.client("ecs", region_name=region)
    
    print("\n=== Waiting for Services to Stabilize ===")
    
    for service in services:
        service_name = f"euglena-{service}"
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


def main():
    """
    Main deployment entry point.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", choices=["gateway", "agent", "all"], default="all",
                       help="Service to deploy (default: all)")
    parser.add_argument("--skip-ecr", action="store_true",
                       help="Skip pushing images to ECR")
    parser.add_argument("--skip-lambda", action="store_true",
                       help="Skip packaging Lambda function")
    parser.add_argument("--skip-network-check", action="store_true",
                       help="Skip network validation")
    parser.add_argument("--wait", action="store_true",
                       help="Wait for services to stabilize after deployment")
    
    args = parser.parse_args()
    
    services_dir = Path.cwd()
    if not (services_dir / "aws.env").exists():
        print("Error: Must run from services/ directory")
        sys.exit(1)
    
    aws_config = load_aws_config(services_dir)
    
    services = ["gateway", "agent"] if args.service == "all" else [args.service]
    
    print("=" * 60)
    print("Euglena Deployment")
    print("=" * 60)
    print(f"Services: {', '.join(services)}")
    print(f"Region: {aws_config['AWS_REGION']}")
    print(f"Cluster: {aws_config['ECS_CLUSTER']}")
    print("=" * 60)
    
    all_success = True
    
    if not args.skip_ecr:
        if not push_to_ecr(services_dir, aws_config, services):
            print("\nFAIL: ECR push failed")
            all_success = False
    else:
        print("\nSKIP: Skipping ECR push")
    
    if not build_and_register_task_definitions(services_dir):
        print("\nFAIL: Task definition build failed")
        all_success = False
    
    if not args.skip_lambda:
        if not package_lambda(services_dir):
            print("\nFAIL: Lambda packaging failed")
            all_success = False
    else:
        print("\nSKIP: Skipping Lambda packaging")
    
    if not args.skip_network_check:
        if not validate_and_fix_network(aws_config):
            print("\nWARN: Network validation had issues (continuing)")
    else:
        print("\nSKIP: Skipping network validation")
    
    if "gateway" in services:
        if not setup_service_discovery(aws_config):
            print("\nWARN: Service discovery setup had issues (continuing)")
        else:
            time.sleep(30)
    
    if not args.skip_lambda:
        if not setup_autoscaling_rule(aws_config):
            print("\nWARN: Autoscaling rule setup had issues (continuing)")
    
    print("\n=== Fixing IAM Permissions ===")
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from fix_iam_permissions import add_task_protection_permission
        if not add_task_protection_permission(aws_config):
            print("  WARN: IAM permission fix had issues (continuing anyway)")
    except Exception as e:
        print(f"  WARN: Could not fix IAM permissions: {e} (continuing anyway)")
    
    if not update_services(services_dir, aws_config, services):
        print("\nFAIL: Service update failed")
        all_success = False
    
    if args.wait:
        time.sleep(120)
        wait_for_services_stable(aws_config, services)
    
    if all_success:
        print("\n" + "=" * 60)
        print("OK: Deployment complete")
        print("=" * 60)
        print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("FAIL: Deployment completed with errors")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
