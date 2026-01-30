#!/usr/bin/env python3
"""
Test runner script for Euglena services.

Builds all containers and runs tests in order:
1. agent-test
2. gateway-test  
3. integration-test

Keeps infrastructure services (rabbitmq, redis, chroma) running between tests
to avoid waiting for RabbitMQ startup (~2 minutes). Only stops test and
application containers between runs.

Stops on first failure and keeps containers running for log inspection.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, List


def get_script_dir() -> Path:
    """
    Get directory containing this script.
    :returns Path: Script directory
    """
    script_path = Path(__file__).parent.resolve()
    if script_path.name == "scripts":
        return script_path.parent / "services"
    return script_path


def run_command(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> int:
    """
    Run a shell command and return exit code.
    :param cmd: Command and arguments as list
    :param cwd: Working directory (defaults to script dir)
    :param check: If True, raise on non-zero exit
    :returns int: Exit code
    """
    if cwd is None:
        cwd = get_script_dir()
    
    result = subprocess.run(cmd, cwd=cwd, check=check)
    return result.returncode


def clear_pytest_cache() -> None:
    """
    Clear pytest cache by removing .pytest_cache directories in test containers.
    :returns None: Nothing is returned
    """
    print("Clearing pytest cache...")
    
    test_services = ["agent-test", "gateway-test", "integration-test"]
    for service in test_services:
        result = subprocess.run(
            ["docker", "compose", "ps", "--format", "json", service],
            cwd=get_script_dir(),
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0 and result.stdout.strip():
            try:
                lines = [l for l in result.stdout.strip().split('\n') if l.strip()]
                if lines:
                    container_info = json.loads(lines[0])
                    container_id = container_info.get('ID') or container_info.get('Names', '').split(',')[0]
                    if container_id:
                        print(f"  Clearing cache in {service}...")
                        subprocess.run(
                            ["docker", "exec", container_id, "sh", "-c", "rm -rf .pytest_cache __pycache__ */__pycache__ */*/__pycache__"],
                            cwd=get_script_dir(),
                            check=False
                        )
            except Exception as e:
                print(f"  Warning: Could not clear cache for {service}: {e}")
    
    print("Cache clearing completed")


def docker_compose_reset_and_rebuild() -> bool:
    """
    Reset everything: stop all containers, remove volumes, prune, and rebuild.
    :returns bool: True if successful
    """
    print("Resetting and rebuilding everything...")
    
    print("  Stopping all containers...")
    run_command(["docker", "compose", "down", "-v"], check=False)
    
    print("  Pruning unused Docker resources...")
    run_command(["docker", "system", "prune", "-f"], check=False)
    
    print("  Building all containers...")
    exit_code = run_command(["docker", "compose", "build", "--no-cache"], check=False)
    if exit_code != 0:
        print("ERROR: Build failed")
        return False
    
    print("Reset and rebuild completed")
    return True


def docker_compose_build() -> bool:
    """
    Build all Docker containers.
    :returns bool: True if successful
    """
    print("Building all containers...")
    
    exit_code = run_command(["docker", "compose", "build"], check=False)
    if exit_code != 0:
        print("ERROR: Build failed")
        return False
    print("Build completed")
    return True


def docker_compose_down_test_services(profile: str = "test", keep_on_failure: bool = False) -> None:
    """
    Stop and remove test containers and application services, but keep infrastructure running.
    :param profile: Docker compose profile to use
    :param keep_on_failure: If True, don't stop containers (for log inspection)
    :returns None: Nothing is returned
    """
    if keep_on_failure:
        print("Keeping containers running for log inspection")
        return
    
    services_to_stop = ["agent-test", "gateway-test", "integration-test", "agent", "gateway", "metrics"]
    
    cmd = ["docker", "compose", "--profile", profile, "stop"] + services_to_stop
    run_command(cmd, check=False)
    
    cmd = ["docker", "compose", "--profile", profile, "rm", "-f"] + services_to_stop
    run_command(cmd, check=False)


def wait_for_rabbitmq(max_wait_seconds: int = 120) -> bool:
    """
    Wait for RabbitMQ to be ready by checking if port 5672 is accessible.
    :param max_wait_seconds: Maximum time to wait in seconds
    :returns bool: True if RabbitMQ is ready
    """
    import socket
    
    print("Waiting for RabbitMQ to be ready...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait_seconds:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('localhost', 5672))
            sock.close()
            if result == 0:
                print("RabbitMQ is ready")
                return True
        except Exception:
            pass
        
        elapsed = int(time.time() - start_time)
        if elapsed % 10 == 0 and elapsed > 0:
            print(f"Still waiting... ({elapsed}s/{max_wait_seconds}s)")
        time.sleep(2)
    
    print(f"ERROR: RabbitMQ not ready after {max_wait_seconds} seconds")
    return False


def docker_compose_ensure_infrastructure() -> bool:
    """
    Ensure infrastructure services (rabbitmq, redis, chroma) are running.
    :returns bool: True if infrastructure is ready
    """
    print("Ensuring infrastructure services are running...")
    
    infrastructure_services = ["rabbitmq", "redis", "chroma"]
    for service in infrastructure_services:
        exit_code = run_command(
            ["docker", "compose", "up", "-d", service],
            check=False
        )
        if exit_code != 0:
            print(f"ERROR: Failed to start {service}")
            return False
    
    if not wait_for_rabbitmq():
        return False
    
    print("Infrastructure services are ready")
    return True


def docker_compose_start_agents(num_agents: int = 2) -> bool:
    """
    Start agent services (scaled to num_agents instances).
    :param num_agents: Number of agent instances to run
    :returns bool: True if agents started successfully
    """
    print(f"Starting {num_agents} agent instance(s)...")
    
    exit_code = run_command(
        ["docker", "compose", "up", "-d", "--scale", f"agent={num_agents}", "agent"],
        check=False
    )
    if exit_code != 0:
        print(f"ERROR: Failed to start agents")
        return False
    
    print(f"Waiting for agents to be ready...")
    time.sleep(10)
    
    print(f"Verifying agent containers are running...")
    result = subprocess.run(
        ["docker", "compose", "ps", "agent", "--format", "json"],
        cwd=get_script_dir(),
        capture_output=True,
        text=True,
        check=False
    )
    if result.returncode == 0:
        containers = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                try:
                    containers.append(json.loads(line))
                except:
                    pass
        running = [c for c in containers if c.get('State') == 'running']
        print(f"Found {len(running)} running agent container(s) (expected {num_agents})")
    
    print(f"Agent services started ({num_agents} instance(s))")
    return True


def docker_compose_up_test(service: str, profile: str = "test") -> bool:
    """
    Run a test service and wait for completion.
    :param service: Service name to run
    :param profile: Docker compose profile to use
    :returns bool: True if tests passed
    """
    print("")
    print("=" * 50)
    print(f"Running {service}...")
    print("=" * 50)
    
    exit_code = run_command(
        ["docker", "compose", "--profile", profile, "up", "--abort-on-container-exit", service],
        check=False
    )
    
    if exit_code != 0:
        print(f"ERROR: {service} failed with exit code {exit_code}")
        print("")
        print("Containers kept running for log inspection")
        print(f"View logs: docker compose --profile {profile} logs {service}")
        print(f"View agent: docker compose logs agent")
        print(f"View gateway: docker compose logs gateway")
        print(f"Clean up: docker compose --profile {profile} down")
        return False
    return True


def run_test_suite(skip_agent: bool = False, skip_gateway: bool = False, skip_integration: bool = False, num_agents: int = 2, clear_cache: bool = True, reset_and_rebuild: bool = False) -> bool:
    """
    Run all tests in order, stopping on first failure.
    Keeps infrastructure services (rabbitmq, redis, chroma) running between tests.
    :param skip_agent: If True, skip agent-test
    :param skip_gateway: If True, skip gateway-test
    :param skip_integration: If True, skip integration-test
    :param num_agents: Number of agent instances to run
    :param clear_cache: If True, clear pytest cache before running tests
    :param reset_and_rebuild: If True, reset everything and rebuild from scratch
    :returns bool: True if all tests passed
    """
    if reset_and_rebuild:
        if not docker_compose_reset_and_rebuild():
            return False
    else:
        if not docker_compose_build():
            return False
    
    if clear_cache:
        clear_pytest_cache()
    
    if not docker_compose_ensure_infrastructure():
        return False
    
    if not skip_gateway or not skip_integration:
        if not docker_compose_start_agents(num_agents):
            return False
    
    test_services = []
    if not skip_agent:
        test_services.append("agent-test")
    if not skip_gateway:
        test_services.append("gateway-test")
    if not skip_integration:
        test_services.append("integration-test")
    
    if not test_services:
        print("WARNING: All tests skipped, nothing to run")
        return True
    
    print(f"Running {len(test_services)} test suite(s): {', '.join(test_services)}")
    
    failed_service = None
    try:
        for service in test_services:
            if not docker_compose_up_test(service):
                failed_service = service
                print(f"ERROR: {service} failed - keeping containers running for log inspection")
                docker_compose_down_test_services(keep_on_failure=True)
                return False
            docker_compose_down_test_services()
        
        print("")
        print("All tests passed")
        return True
    except KeyboardInterrupt:
        print("\nInterrupted by user - keeping containers running")
        docker_compose_down_test_services(keep_on_failure=True)
        return False
    except Exception as e:
        print(f"\nERROR: Unexpected error: {e}")
        if failed_service:
            print(f"Failed service was: {failed_service}")
        docker_compose_down_test_services(keep_on_failure=True)
        return False


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    :returns Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run Euglena service tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --skip-agent        # Skip agent-test
  python run_tests.py --skip-gateway      # Skip gateway-test
  python run_tests.py --skip-integration  # Skip integration-test
  python run_tests.py --skip-agent --skip-gateway  # Run only integration-test
  python run_tests.py --num-agents 3      # Run with 3 agent instances
  python run_tests.py --no-clear-cache    # Skip cache clearing
        """
    )
    parser.add_argument(
        "--skip-agent",
        action="store_true",
        help="Skip agent-test suite"
    )
    parser.add_argument(
        "--skip-gateway",
        action="store_true",
        help="Skip gateway-test suite"
    )
    parser.add_argument(
        "--skip-integration",
        action="store_true",
        help="Skip integration-test suite"
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=2,
        help="Number of agent instances to run (default: 2)"
    )
    parser.add_argument(
        "--no-clear-cache",
        action="store_true",
        help="Skip clearing pytest cache before running tests"
    )
    parser.add_argument(
        "--reset-and-rebuild",
        action="store_true",
        help="Reset everything (stop all containers, remove volumes, prune, rebuild from scratch)"
    )
    return parser.parse_args()


def main() -> int:
    """
    Main entry point.
    :returns int: Exit code (0 = success, 1 = failure)
    """
    args = parse_args()
    
    try:
        success = run_test_suite(
            skip_agent=args.skip_agent,
            skip_gateway=args.skip_gateway,
            skip_integration=args.skip_integration,
            num_agents=args.num_agents,
            clear_cache=not args.no_clear_cache,
            reset_and_rebuild=args.reset_and_rebuild
        )
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        docker_compose_down_test_services()
        return 130
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        docker_compose_down_test_services()
        return 1


if __name__ == "__main__":
    sys.exit(main())
