"""
Package the autoscaling Lambda from services/lambda_autoscaling.
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Tuple


def run_command(cmd, check: bool = True, capture: bool = False) -> Tuple[bool, str, str]:
    """
    Run a shell command and return result.

    :param cmd: Command as list of strings
    :param check: Whether to raise on non-zero exit
    :param capture: Whether to capture output
    :returns: Tuple of (success, stdout, stderr)
    """
    try:
        result = subprocess.run(cmd, check=check, capture_output=capture, text=True)
        return True, result.stdout if capture else "", result.stderr if capture else ""
    except subprocess.CalledProcessError as e:
        return False, e.stdout if capture else "", e.stderr if capture else str(e)
    except FileNotFoundError:
        return False, "", f"Command not found: {cmd[0]}"


def package_lambda() -> Path:
    """
    Package Lambda function for deployment.

    :returns: Path to the created zip package
    """
    repo_root = Path(__file__).resolve().parent.parent
    services_dir = repo_root / "services"
    lambda_dir = services_dir / "lambda_autoscaling"
    output_dir = repo_root / "dist"

    if not lambda_dir.exists():
        raise FileNotFoundError(f"Lambda directory not found: {lambda_dir}")

    output_dir.mkdir(exist_ok=True)
    package_name = f"lambda-autoscaling-{datetime.now().strftime('%Y%m%d%H%M%S')}.zip"
    package_path = output_dir / package_name

    temp_dir = output_dir / "lambda-package"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

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
        success, _, stderr = run_command(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements), "-t", str(temp_dir), "--quiet"],
            check=False,
            capture=True,
        )
        if not success and stderr:
            print(f"Dependency installation issues: {stderr}")

    with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(temp_dir)
                zipf.write(file_path, arcname)

    shutil.rmtree(temp_dir)
    return package_path


def parse_args():
    """
    Parse CLI arguments.

    :returns: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Package autoscaling Lambda")
    return parser.parse_args()


def main() -> None:
    """
    Main entry point.

    :returns: None
    """
    args = parse_args()
    _ = args
    package_path = package_lambda()
    print(f"Lambda package created: {package_path}")


if __name__ == "__main__":
    main()
