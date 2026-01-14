#!/usr/bin/env python3
"""
Create Lambda deployment package for autoscaling function.

Packages lambda_function.py, env files, and dependencies into a ZIP file.
Expected to be run from project root: python scripts/package-lambda.py
"""

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from datetime import datetime


def main():
    """Create Lambda deployment package."""
    project_root = Path(__file__).parent.parent
    services_dir = project_root / "services"
    lambda_dir = services_dir / "lambda_autoscaling"
    output_dir = project_root / "dist"
    package_name = f"lambda-autoscaling-{datetime.now().strftime('%Y%m%d%H%M%S')}.zip"
    
    if not lambda_dir.exists():
        print(f"Error: {lambda_dir} not found")
        sys.exit(1)
    
    output_dir.mkdir(exist_ok=True)
    package_path = output_dir / package_name
    
    if package_path.exists():
        package_path.unlink()
    
    temp_dir = output_dir / "lambda-package"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    
    print("Copying lambda_function.py...")
    shutil.copy(lambda_dir / "lambda_function.py", temp_dir / "lambda_function.py")
    
    print("Copying env files...")
    shutil.copy(services_dir / "aws.env", temp_dir / "aws.env")
    shutil.copy(services_dir / ".env", temp_dir / ".env")
    
    print("Installing dependencies...")
    requirements = lambda_dir / "requirements.txt"
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "-r", str(requirements),
        "-t", str(temp_dir),
        "--quiet"
    ], check=True)
    
    print(f"Creating ZIP package: {package_path}")
    with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(temp_dir)
                zipf.write(file_path, arcname)
    
    print(f"Cleaning up temp directory...")
    shutil.rmtree(temp_dir)
    
    size_mb = package_path.stat().st_size / (1024 * 1024)
    print(f"Package created: {package_path} ({size_mb:.2f} MB)")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
