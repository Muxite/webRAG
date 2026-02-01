"""
Task definition building for deployment scripts.
"""
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
    print("\n=== Building Task Definitions ===")
    
    script_path = services_dir.parent / "scripts" / "build-task-definition.py"
    cmd = ["python", str(script_path), "--mode", str(mode.value)]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1
    )
    for line in process.stdout:
        try:
            safe_line = line.rstrip().encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8', errors='replace')
            print(safe_line)
        except (UnicodeEncodeError, UnicodeDecodeError):
            safe_line = line.rstrip().encode('ascii', errors='replace').decode('ascii')
            print(safe_line)
    process.wait()
    
    if process.returncode != 0:
        print(f"  FAIL: Failed to build task definitions (exit code {process.returncode})")
        return False
    
    print("  OK: Task definitions built and registered")
    return True
