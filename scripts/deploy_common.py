"""
Common utilities for deployment scripts.
"""
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
            encoding='utf-8',
            errors='replace',
            shell=shell
        )
        return True, result.stdout if capture else "", result.stderr if capture else ""
    except subprocess.CalledProcessError as e:
        return False, e.stdout if capture else "", e.stderr if capture else str(e)
    except FileNotFoundError:
        return False, "", f"Command not found: {cmd[0] if isinstance(cmd, list) else cmd}"


def get_image_size(image_name: str) -> Optional[str]:
    """
    Get the size of a Docker image in human-readable format.
    
    :param image_name: Docker image name (e.g., "euglena/agent").
    :returns: Formatted size string (e.g., "245.3 MB") or None on error.
    """
    try:
        cmd = ["docker", "inspect", "--format", "{{.Size}}", image_name]
        success, stdout, stderr = run_command(cmd, check=False, capture=True)
        if not success or not stdout.strip():
            return None
        
        size_bytes = int(stdout.strip())
        
        if size_bytes >= 1024 * 1024 * 1024:
            size_gb = size_bytes / (1024 * 1024 * 1024)
            return f"{size_gb:.2f} GB"
        elif size_bytes >= 1024 * 1024:
            size_mb = size_bytes / (1024 * 1024)
            return f"{size_mb:.2f} MB"
        elif size_bytes >= 1024:
            size_kb = size_bytes / 1024
            return f"{size_kb:.2f} KB"
        else:
            return f"{size_bytes} B"
    except (ValueError, Exception):
        return None
