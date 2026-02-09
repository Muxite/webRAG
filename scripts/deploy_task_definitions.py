"""
Task definition building for deployment scripts.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from deploy_common import run_command

try:
    from scripts.deployment_mode import DeploymentMode
except ImportError:
    from deployment_mode import DeploymentMode


def build_and_register_task_definitions(services_dir: Path, mode: DeploymentMode = DeploymentMode.SINGLE) -> bool:
    """
    Build and register task definitions.
    :param services_dir: Services directory path.
    :param mode: Deployment mode enum.
    :returns: True on success.
    """
    print("\nBuild task definitions")
    
    script_path = services_dir.parent / "scripts" / "build_task_definition.py"
    cmd = ["python", str(script_path), "--mode", str(mode.value)]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=1
    )
    if process.stdout:
        for line in iter(process.stdout.readline, b""):
            try:
                decoded = line.decode(sys.stdout.encoding or "utf-8", errors="replace").rstrip()
                print(decoded)
            except Exception:
                print(line.decode("utf-8", errors="replace").rstrip())
    process.wait()
    
    if process.returncode != 0:
        print(f"Build task definitions failed: exit {process.returncode}")
        return False
    
    print("Task definitions built and registered")
    return True


def parse_args():
    """
    Parse CLI arguments.
    :returns: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Build and register task definitions")
    parser.add_argument("--services-dir", type=Path, default=None,
                       help="Services directory containing aws.env")
    parser.add_argument("--mode", choices=["single", "autoscale"], default="single",
                       help="Deployment mode")
    return parser.parse_args()


def main():
    """
    Run the task definition build workflow.
    :returns: None.
    """
    args = parse_args()
    services_dir = args.services_dir or Path.cwd()
    mode = DeploymentMode.from_string(args.mode)
    success = build_and_register_task_definitions(services_dir, mode)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

