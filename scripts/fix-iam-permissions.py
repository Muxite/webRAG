"""
Add IAM permissions for ECS task protection and EFS access.
"""
import boto3
import sys
import json
from pathlib import Path
from dotenv import dotenv_values


def load_aws_config(services_dir: Path) -> dict:
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


def get_role_policies(iam_client, role_name: str) -> dict:
    """
    Get all inline policies for a role and check for task protection permission.
    
    :param iam_client: Boto3 IAM client.
    :param role_name: IAM role name.
    :returns: Policy document dictionary with existing statements.
    """
    policy_doc = {
        "Version": "2012-10-17",
        "Statement": []
    }
    
    try:
        response = iam_client.list_role_policies(RoleName=role_name)
        policy_names = response.get("PolicyNames", [])
        
        for policy_name in policy_names:
            try:
                policy_response = iam_client.get_role_policy(
                    RoleName=role_name,
                    PolicyName=policy_name
                )
                existing_doc = json.loads(policy_response["PolicyDocument"])
                if "Statement" in existing_doc:
                    for stmt in existing_doc["Statement"]:
                        if isinstance(stmt.get("Action"), list):
                            actions = stmt["Action"]
                        elif isinstance(stmt.get("Action"), str):
                            actions = [stmt["Action"]]
                        else:
                            continue
                        
                        if "ecs:UpdateTaskProtection" in actions:
                            return existing_doc
            except Exception:
                continue
    except Exception as e:
        print(f"  WARN: Could not list policies: {e}")
    
    return policy_doc


def add_efs_permissions(aws_config: dict) -> bool:
    """
    Add EFS permissions to ecsTaskExecutionRole for mounting EFS volumes.
    
    :param aws_config: AWS configuration dictionary.
    :returns: True on success.
    """
    print("\n=== Adding IAM Permissions for EFS Access ===")
    
    region = aws_config.get("AWS_REGION", "us-east-1")
    role_name = aws_config.get("ECS_TASK_EXECUTION_ROLE_NAME", "ecsTaskExecutionRole")
    efs_file_system_id = aws_config.get("EFS_FILE_SYSTEM_ID", "").strip()
    
    if not efs_file_system_id:
        print("  SKIP: EFS_FILE_SYSTEM_ID not set (EFS not configured)")
        return True
    
    iam_client = boto3.client("iam", region_name=region)
    
    try:
        print(f"  Role name: {role_name}")
        
        try:
            iam_client.get_role(RoleName=role_name)
            print(f"  OK: Role exists: {role_name}")
        except iam_client.exceptions.NoSuchEntityException:
            print(f"  FAIL: Role not found: {role_name}")
            return False
        
        efs_file_system_arn = f"arn:aws:elasticfilesystem:{region}:{aws_config.get('AWS_ACCOUNT_ID', '')}:file-system/{efs_file_system_id}"
        
        required_permissions = {
            "Effect": "Allow",
            "Action": [
                "elasticfilesystem:ClientMount",
                "elasticfilesystem:ClientWrite",
                "elasticfilesystem:ClientRootAccess",
                "elasticfilesystem:DescribeMountTargets"
            ],
            "Resource": efs_file_system_arn
        }
        
        try:
            response = iam_client.list_role_policies(RoleName=role_name)
            policy_names = response.get("PolicyNames", [])
            
            efs_policy_name = "EFSAccessPolicy"
            has_efs_policy = efs_policy_name in policy_names
            
            if has_efs_policy:
                existing_policy = iam_client.get_role_policy(
                    RoleName=role_name,
                    PolicyName=efs_policy_name
                )
                existing_doc = json.loads(existing_policy["PolicyDocument"])
                
                has_permissions = False
                for stmt in existing_doc.get("Statement", []):
                    actions = stmt.get("Action", [])
                    if isinstance(actions, str):
                        actions = [actions]
                    if (stmt.get("Effect") == "Allow" and 
                        "elasticfilesystem:ClientMount" in actions):
                        has_permissions = True
                        break
                
                if has_permissions:
                    print(f"  OK: EFS permissions already exist")
                    return True
            
            policy_doc = {
                "Version": "2012-10-17",
                "Statement": [required_permissions]
            }
            
            print("  Adding EFS permissions...")
            iam_client.put_role_policy(
                RoleName=role_name,
                PolicyName=efs_policy_name,
                PolicyDocument=json.dumps(policy_doc)
            )
            print(f"  OK: EFS permissions added successfully!")
            return True
            
        except Exception as e:
            print(f"  WARN: Could not add EFS permissions: {e}")
            print(f"  NOTE: You may need to manually add EFS permissions to {role_name}")
            return False
    
    except Exception as e:
        print(f"  WARN: Error checking EFS permissions: {e}")
        return False


def add_task_protection_permission(aws_config: dict) -> bool:
    """
    Add ecs:UpdateTaskProtection permission to ecsTaskRole.
    
    :param aws_config: AWS configuration dictionary.
    :returns: True on success.
    """
    print("\n=== Adding IAM Permission for Task Protection ===")
    
    region = aws_config.get("AWS_REGION", "us-east-1")
    role_name = aws_config.get("ECS_TASK_ROLE_NAME", "ecsTaskRole")
    
    iam_client = boto3.client("iam", region_name=region)
    
    try:
        print(f"  Role name: {role_name}")
        
        try:
            iam_client.get_role(RoleName=role_name)
            print(f"  OK: Role exists: {role_name}")
        except iam_client.exceptions.NoSuchEntityException:
            print(f"  FAIL: Role not found: {role_name}")
            print("  Create the role first or update ECS_TASK_ROLE_NAME in aws.env")
            return False
        
        policy_doc = get_role_policies(iam_client, role_name)
        
        statements = policy_doc.get("Statement", [])
        
        has_permission = False
        for stmt in statements:
            actions = stmt.get("Action", [])
            if isinstance(actions, str):
                actions = [actions]
            
            if (stmt.get("Effect") == "Allow" and 
                "ecs:UpdateTaskProtection" in actions):
                has_permission = True
                print(f"  OK: Permission already exists in role")
                break
        
        if not has_permission:
            required_permission = {
                "Effect": "Allow",
                "Action": "ecs:UpdateTaskProtection",
                "Resource": "*"
            }
            
            if not isinstance(statements, list):
                statements = []
            
            statements.append(required_permission)
            policy_doc["Statement"] = statements
            
            policy_json = json.dumps(policy_doc)
            
            print("  Adding permission...")
            iam_client.put_role_policy(
                RoleName=role_name,
                PolicyName="TaskProtectionPolicy",
                PolicyDocument=policy_json
            )
            print(f"  OK: Permission added successfully!")
        
        return True
    
    except Exception as e:
        print(f"  FAIL: Error adding permission: {e}")
        import traceback
        traceback.print_exc()
        return False


def add_cloudwatch_permissions(aws_config: dict) -> bool:
    """
    Add CloudWatch PutMetricData permission to ecsTaskRole for metrics service.
    
    :param aws_config: AWS configuration dictionary.
    :returns: True on success.
    """
    print("\n=== Adding IAM Permission for CloudWatch Metrics ===")
    
    region = aws_config.get("AWS_REGION", "us-east-1")
    role_name = aws_config.get("ECS_TASK_ROLE_NAME", "ecsTaskRole")
    
    iam_client = boto3.client("iam", region_name=region)
    
    try:
        print(f"  Role name: {role_name}")
        
        try:
            iam_client.get_role(RoleName=role_name)
            print(f"  OK: Role exists: {role_name}")
        except iam_client.exceptions.NoSuchEntityException:
            print(f"  FAIL: Role not found: {role_name}")
            print("  Create the role first or update ECS_TASK_ROLE_NAME in aws.env")
            return False
        
        policy_doc = get_role_policies(iam_client, role_name)
        
        statements = policy_doc.get("Statement", [])
        
        has_permission = False
        for stmt in statements:
            actions = stmt.get("Action", [])
            if isinstance(actions, str):
                actions = [actions]
            
            if (stmt.get("Effect") == "Allow" and 
                "cloudwatch:PutMetricData" in actions):
                has_permission = True
                print(f"  OK: CloudWatch permission already exists in role")
                break
        
        if not has_permission:
            required_permission = {
                "Effect": "Allow",
                "Action": "cloudwatch:PutMetricData",
                "Resource": "*"
            }
            
            if not isinstance(statements, list):
                statements = []
            
            statements.append(required_permission)
            policy_doc["Statement"] = statements
            
            policy_json = json.dumps(policy_doc)
            
            print("  Adding CloudWatch permission...")
            iam_client.put_role_policy(
                RoleName=role_name,
                PolicyName="CloudWatchMetricsPolicy",
                PolicyDocument=policy_json
            )
            print(f"  OK: CloudWatch permission added successfully!")
        
        return True
    
    except Exception as e:
        print(f"  WARN: Error adding CloudWatch permission: {e}")
        return False


def main():
    """
    Main entry point.
    """
    services_dir = Path.cwd()
    if not (services_dir / "aws.env").exists():
        print(f"Error: aws.env not found in current directory ({services_dir})")
        print("  Please run this script from the services/ directory")
        sys.exit(1)
    
    aws_config = load_aws_config(services_dir)
    
    efs_success = add_efs_permissions(aws_config)
    task_protection_success = add_task_protection_permission(aws_config)
    cloudwatch_success = add_cloudwatch_permissions(aws_config)
    
    if efs_success and task_protection_success and cloudwatch_success:
        print("\nOK: IAM permission update completed")
    else:
        print("\nWARN: Some IAM permission updates had issues (check output above)")
        if not task_protection_success:
            sys.exit(1)


if __name__ == "__main__":
    main()
