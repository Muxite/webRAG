"""
Create or update AWS Secrets Manager secret from keys.env.

Reads secrets from keys.env and stores them as a JSON object in AWS Secrets Manager.
Expected to be run from services/ directory: python ../scripts/create-secrets.py
"""

import json
import sys
from pathlib import Path
from dotenv import dotenv_values
import boto3
from botocore.exceptions import ClientError


def load_aws_config(services_dir):
    """
    Load AWS configuration from aws.env file.
    
    :param services_dir: Services directory path
    :return: dict with region, secret_name
    """
    aws_env_path = services_dir / "aws.env"
    if not aws_env_path.exists():
        print(f"Error: {aws_env_path} not found")
        return None
    
    aws_env = dict(dotenv_values(str(aws_env_path)))
    region = aws_env.get("AWS_REGION", "").strip()
    secret_name = aws_env.get("AWS_SECRET_NAME", "").strip()
    
    if not region:
        print("Error: AWS_REGION not found in aws.env")
        return None
    if not secret_name:
        print("Error: AWS_SECRET_NAME not found in aws.env")
        return None
    
    return {"region": region, "secret_name": secret_name}


def load_secrets(services_dir):
    """
    Load secrets from keys.env file.
    
    :param services_dir: Services directory path
    :return: Dictionary of secret key-value pairs
    """
    keys_env_path = services_dir / "keys.env"
    if not keys_env_path.exists():
        print(f"Error: {keys_env_path} not found")
        return None
    
    keys_env = dict(dotenv_values(str(keys_env_path)))
    secrets = {k: v for k, v in keys_env.items() if k and v and not k.startswith("#")}
    
    if not secrets:
        print("Error: No secrets found in keys.env")
        return None
    
    return secrets


def create_or_update_secret(secret_name, secrets_dict, region):
    """
    Create or update secret in AWS Secrets Manager.
    
    :param secret_name: Secret name in Secrets Manager
    :param secrets_dict: Dictionary of secret key-value pairs
    :param region: AWS region
    :return: True on success, False on failure
    """
    client = boto3.client('secretsmanager', region_name=region)
    secret_string = json.dumps(secrets_dict)
    
    try:
        client.describe_secret(SecretId=secret_name)
        client.update_secret(
            SecretId=secret_name,
            SecretString=secret_string
        )
        print(f"Updated secret: {secret_name}")
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            client.create_secret(
                Name=secret_name,
                SecretString=secret_string
            )
            print(f"Created secret: {secret_name}")
        else:
            print(f"Error: {e}")
            return False
    
    return True


def main():
    """Main entry point."""
    services_dir = Path.cwd()
    
    aws_config = load_aws_config(services_dir)
    if not aws_config:
        sys.exit(1)
    
    secrets = load_secrets(services_dir)
    if not secrets:
        sys.exit(1)
    
    print(f"Found {len(secrets)} secrets")
    
    if not create_or_update_secret(
        aws_config["secret_name"],
        secrets,
        aws_config["region"]
    ):
        sys.exit(1)
    
    from datetime import datetime
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
