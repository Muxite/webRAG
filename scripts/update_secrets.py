"""
Update Secrets Manager secret from keys.env.
"""
import argparse
import sys
from pathlib import Path


def parse_args():
    """
    Parse CLI arguments.
    :returns: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Update Secrets Manager from keys.env")
    parser.add_argument("--secret-name", help="Secret name (overrides aws.env)")
    return parser.parse_args()


def main():
    """
    Update Secrets Manager from keys.env.
    :returns: None.
    """
    script_dir = Path(__file__).parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from register_secrets import update_secrets_from_keys_env
    args = parse_args()
    services_dir = Path.cwd()
    if not (services_dir / "aws.env").exists():
        print(f"aws.env not found in {services_dir}")
        print("Run from services/")
        sys.exit(1)
    success = update_secrets_from_keys_env(services_dir, secret_name_override=args.secret_name)
    if success:
        print("Secrets updated")
        sys.exit(0)
    print("Secrets update failed")
    sys.exit(1)


if __name__ == "__main__":
    main()
