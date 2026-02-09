"""
Register secrets in AWS Secrets Manager and get the ARN suffix.
"""
import boto3
import json
import sys
import argparse
from pathlib import Path
from dotenv import dotenv_values


def load_aws_config(services_dir: Path) -> dict:
    """
    Load AWS configuration from aws.env.
    :param services_dir: Services directory path.
    :returns: Configuration dictionary.
    """
    aws_env_path = services_dir / "aws.env"
    if not aws_env_path.exists():
        print(f"Error: {aws_env_path} not found")
        sys.exit(1)
    
    return dict(dotenv_values(str(aws_env_path)))


def load_keys_env(services_dir: Path) -> dict:
    """
    Load secrets from keys.env.
    :param services_dir: Services directory path.
    :returns: Dictionary of keys.
    """
    keys_env_path = services_dir / "keys.env"
    if not keys_env_path.exists():
        print(f"Error: {keys_env_path} not found")
        sys.exit(1)
    
    return dict(dotenv_values(str(keys_env_path)))


def extract_arn_suffix(secret_arn: str) -> str:
    """
    Extract the suffix portion of a secret ARN.
    :param secret_arn: Full secret ARN.
    :returns: ARN suffix.
    """
    if not secret_arn.startswith("arn:aws:secretsmanager:"):
        return secret_arn
    
    secret_idx = secret_arn.find("secret:")
    if secret_idx == -1:
        return secret_arn
    
    after_secret = secret_arn[secret_idx + 7:]
    name_suffix_part = after_secret.split(":")[0] if ":" in after_secret else after_secret
    
    if "-" in name_suffix_part:
        return name_suffix_part[name_suffix_part.rfind("-") + 1:]
    
    return secret_arn


def create_or_update_secret(aws_config: dict, keys: dict, create: bool = False) -> tuple[bool, str]:
    """
    Create or update the Secrets Manager entry.
    :param aws_config: AWS configuration dictionary.
    :param keys: Dictionary of keys to store.
    :param create: If True, create new secret; if False, update existing.
    :returns: Tuple of (success, arn_suffix).
    """
    region = aws_config["AWS_REGION"]
    secret_name = aws_config.get("AWS_SECRET_NAME", "euglena-secrets")
    
    secrets_client = boto3.client("secretsmanager", region_name=region)
    
    secret_string = json.dumps(keys, indent=2)
    
    print(f"\nSecret {('create' if create else 'update')}")
    print(f"Name: {secret_name}")
    print(f"Region: {region}")
    print(f"Keys: {', '.join(keys.keys())}")
    
    try:
        if create:
            try:
                response = secrets_client.create_secret(
                    Name=secret_name,
                    SecretString=secret_string,
                    Description="Euglena service secrets"
                )
                print("Secret created")
            except secrets_client.exceptions.ResourceExistsException:
                print("Secret exists, updating")
                response = secrets_client.update_secret(
                    SecretId=secret_name,
                    SecretString=secret_string
                )
                print("Secret updated")
        else:
            response = secrets_client.update_secret(
                SecretId=secret_name,
                SecretString=secret_string
            )
            print("Secret updated")
        
        arn = response.get("ARN", "")
        if arn:
            suffix = extract_arn_suffix(arn)
            print(f"\nARN: {arn}")
            print(f"Suffix: {suffix}")
            print(f"\naws.env: AWS_SECRET_ARN_SUFFIX={suffix}")
            return True, suffix
        else:
            print("No ARN in response")
            return True, ""
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False, ""


def update_secrets_from_keys_env(services_dir: Path, secret_name_override: str | None = None) -> bool:
    """
    Update Secrets Manager from keys.env.
    :param services_dir: Services directory path.
    :param secret_name_override: Optional secret name override.
    :returns: True on success.
    """
    aws_config = load_aws_config(services_dir)
    if secret_name_override:
        aws_config["AWS_SECRET_NAME"] = secret_name_override
    keys = load_keys_env(services_dir)
    if not keys:
        print("Error: No keys found in keys.env")
        return False
    success, _ = create_or_update_secret(aws_config, keys, create=False)
    return success


def get_existing_secret_arn(aws_config: dict) -> tuple[bool, str]:
    """
    Fetch the ARN suffix for an existing secret.
    :param aws_config: AWS configuration dictionary.
    :returns: Tuple of (success, arn_suffix).
    """
    region = aws_config["AWS_REGION"]
    secret_name = aws_config.get("AWS_SECRET_NAME", "euglena-secrets")
    
    secrets_client = boto3.client("secretsmanager", region_name=region)
    
    print("\nSecret ARN lookup")
    print(f"Name: {secret_name}")
    print(f"Region: {region}")
    
    try:
        response = secrets_client.describe_secret(SecretId=secret_name)
        arn = response.get("ARN", "")
        
        if arn:
            suffix = extract_arn_suffix(arn)
            print(f"\nARN: {arn}")
            print(f"Suffix: {suffix}")
            print(f"\naws.env: AWS_SECRET_ARN_SUFFIX={suffix}")
            return True, suffix
        else:
            print("No ARN found")
            return False, ""
    except secrets_client.exceptions.ResourceNotFoundException:
        print(f"Secret not found: {secret_name}")
        return False, ""
    except Exception as e:
        print(f"Error: {e}")
        return False, ""


def parse_args():
    """
    Parse CLI arguments.
    :returns: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Register secrets in AWS Secrets Manager")
    parser.add_argument("--create", action="store_true", help="Create new secret")
    parser.add_argument("--update", action="store_true", help="Update existing secret")
    parser.add_argument("--get-arn", action="store_true", help="Get ARN of existing secret")
    parser.add_argument("--secret-name", help="Secret name (overrides aws.env)")
    
    return parser.parse_args()

def main():
    """
    Run the secrets registration workflow.
    :returns: None.
    """
    args = parse_args()
    
    services_dir = Path.cwd()
    if not (services_dir / "aws.env").exists():
        print(f"aws.env not found in {services_dir}")
        print("Run from services/")
        sys.exit(1)
    
    aws_config = load_aws_config(services_dir)
    
    if args.secret_name:
        aws_config["AWS_SECRET_NAME"] = args.secret_name
    
    if args.get_arn:
        success, suffix = get_existing_secret_arn(aws_config)
        sys.exit(0 if success else 1)
    
    keys = load_keys_env(services_dir)
    
    if not keys:
        print("Error: No keys found in keys.env")
        sys.exit(1)
    
    create = args.create or (not args.update)
    success, suffix = create_or_update_secret(aws_config, keys, create=create)
    
    if success:
        print("\nOK: Secret operation completed")
        if suffix:
            print(f"\nRemember to add AWS_SECRET_ARN_SUFFIX={suffix} to your aws.env")
    else:
        print("\nFAIL: Secret operation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
