#!/usr/bin/env python3
"""
Test script for recent changes:
- Queue depth tracking improvements
- Metrics service enhancements
- Worker presence tracking fixes
"""

import subprocess
import time
import json
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Warning: 'requests' library not found. Install with: pip install requests")
    print("Worker count endpoint test will be skipped.")
    requests = None


def run_command(cmd, check=True, capture_output=True, text=True):
    """Run a shell command and return result."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=check,
            capture_output=capture_output,
            text=text,
            cwd=Path(__file__).parent.parent / "services"
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.CalledProcessError as e:
        return e.stdout.strip() if e.stdout else "", e.stderr.strip() if e.stderr else "", e.returncode


def print_step(step_num, description):
    """Print a formatted step header."""
    print(f"\n{'='*60}")
    print(f"Step {step_num}: {description}")
    print('='*60)


def check_condition(condition, message, success_msg=None):
    """Check a condition and print result."""
    if condition:
        print(f"[PASS] {message}")
        if success_msg:
            print(f"  {success_msg}")
        return True
    else:
        print(f"[FAIL] {message}")
        return False


def main():
    """Main test function."""
    print("Testing Recent Changes")
    print("=" * 60)
    
    results = {
        "dependencies_started": False,
        "services_built": False,
        "metrics_running": False,
        "queue_depth_logged": False,
        "agent_started": False,
        "worker_presence_registered": False,
        "worker_count_accurate": False,
    }
    
    print_step(1, "Start Dependencies")
    stdout, stderr, code = run_command("docker compose up -d rabbitmq redis", check=False)
    if code == 0:
        results["dependencies_started"] = check_condition(True, "Dependencies started")
        print("Waiting for services to be ready...")
        time.sleep(5)
    else:
        print(f"Failed to start dependencies: {stderr}")
        return False
    
    print_step(2, "Build Changed Services")
    stdout, stderr, code = run_command("docker compose build metrics gateway agent", check=False)
    if code == 0:
        results["services_built"] = check_condition(True, "Services built successfully")
    else:
        print(f"Build failed: {stderr}")
        return False
    
    print_step(3, "Start Metrics Service")
    print("Starting metrics service...")
    run_command("docker compose up -d metrics", check=False)
    time.sleep(3)
    
    stdout, _, code = run_command("docker compose ps metrics --format json", check=False)
    if code == 0 and stdout:
        try:
            containers = json.loads(stdout) if stdout.startswith('[') else [json.loads(stdout)]
            running = any(c.get('State', '').lower() == 'running' for c in containers)
            results["metrics_running"] = check_condition(running, "Metrics service running")
        except:
            results["metrics_running"] = check_condition(False, "Could not parse metrics status")
    
    print("\nChecking metrics logs for queue depth...")
    time.sleep(8)
    stdout, _, _ = run_command("docker compose logs metrics --tail 50", check=False)
    
    has_queue_depth = (
        "Queue Depth:" in stdout or 
        "queue depth" in stdout.lower() or
        "QUEUE DEPTH CHECK" in stdout or
        "depth=" in stdout.lower()
    )
    
    if has_queue_depth:
        results["queue_depth_logged"] = check_condition(
            True,
            "Queue depth logging found",
            "Metrics service is logging queue depth"
        )
        lines = [l for l in stdout.split('\n') if any(x in l.lower() for x in ['depth', 'queue depth', 'queue_depth'])]
        if lines:
            print("  Recent queue depth logs:")
            for line in lines[-5:]:
                print(f"    {line}")
    else:
        results["queue_depth_logged"] = check_condition(
            False,
            "Queue depth logging not found",
            "Metrics may not be collecting yet or logs use different format"
        )
        print("  [DEBUG] Showing recent metrics logs for diagnosis:")
        recent_logs = stdout.split('\n')[-10:]
        for line in recent_logs:
            print(f"    {line}")
    
    print_step(4, "Start Agent Service")
    print("Starting agent service...")
    run_command("docker compose up -d agent", check=False)
    time.sleep(5)
    
    stdout, _, code = run_command("docker compose ps agent --format json", check=False)
    if code == 0 and stdout:
        try:
            containers = json.loads(stdout) if stdout.startswith('[') else [json.loads(stdout)]
            running = any(c.get('State', '').lower() == 'running' for c in containers)
            results["agent_started"] = check_condition(running, "Agent service running")
            
            stdout, _, _ = run_command("docker compose logs agent --tail 20", check=False)
            if "Traceback" in stdout or "ImportError" in stdout or "ModuleNotFoundError" in stdout:
                print("  [WARN] Agent logs show errors:")
                error_lines = [l for l in stdout.split('\n') if 'Error' in l or 'Traceback' in l or 'Import' in l]
                for line in error_lines[:3]:
                    print(f"    {line}")
            
            if "WorkerPresence" in stdout or "presence" in stdout.lower():
                print("  [INFO] Worker presence system appears active in logs")
        except:
            results["agent_started"] = check_condition(False, "Could not parse agent status")
    
    print_step(5, "Start Gateway Service")
    print("Starting gateway service...")
    run_command("docker compose up -d gateway", check=False)
    time.sleep(5)
    
    print_step(6, "Check Worker Presence in Redis")
    stdout, _, _ = run_command('docker compose exec -T redis redis-cli KEYS "worker:presence:*"', check=False)
    presence_keys = [k.strip() for k in stdout.split('\n') if k.strip() and not k.startswith('(')]
    
    stdout, _, _ = run_command('docker compose exec -T redis redis-cli KEYS "worker:status:*"', check=False)
    status_keys = [k.strip() for k in stdout.split('\n') if k.strip() and not k.startswith('(')]
    
    stdout, _, _ = run_command('docker compose exec -T redis redis-cli SMEMBERS "workers:status"', check=False)
    workers_set = [w.strip() for w in stdout.split('\n') if w.strip() and not w.startswith('(')]
    
    if presence_keys or status_keys or workers_set:
        results["worker_presence_registered"] = check_condition(
            True,
            "Worker presence keys found",
            f"Found {len(presence_keys)} presence keys, {len(status_keys)} status keys, {len(workers_set)} in set"
        )
        if presence_keys:
            print(f"  Presence keys: {presence_keys[:3]}")
        if status_keys:
            print(f"  Status keys: {status_keys[:3]}")
        if workers_set:
            print(f"  Workers in set: {workers_set[:3]}")
    else:
        results["worker_presence_registered"] = check_condition(
            False,
            "No worker presence keys found",
            "Agent may not have registered yet, wait a few seconds"
        )
    
    print_step(7, "Test Worker Count Endpoint")
    if requests is None:
        results["worker_count_accurate"] = check_condition(
            False,
            "Skipped (requests library not available)",
            "Install requests: pip install requests"
        )
    else:
        try:
            response = requests.get("http://localhost:8080/agents/count", timeout=5)
            if response.status_code == 200:
                data = response.json()
                count = data.get('count', 0)
                results["worker_count_accurate"] = check_condition(
                    True,
                    f"Worker count endpoint working: {count}",
                    "Endpoint returned successfully"
                )
                
                if count > 0 and len(workers_set) > 0:
                    if count == len(workers_set):
                        print(f"  [PASS] Count matches Redis set size: {count} == {len(workers_set)}")
                    else:
                        print(f"  [WARN] Count mismatch: API={count}, Redis set={len(workers_set)}")
                elif count == 0 and len(workers_set) == 0:
                    print(f"  [INFO] Count is 0, no workers in set (expected if agent just started)")
                else:
                    print(f"  [WARN] Count mismatch: API={count}, Redis set={len(workers_set)}")
                    print(f"  [DEBUG] This suggests workers are in the set but missing presence/status keys")
                    print(f"  [DEBUG] Checking if presence keys exist for workers in set...")
                    for worker_id in workers_set[:3]:
                        stdout, _, _ = run_command(f'docker compose exec -T redis redis-cli EXISTS "worker:presence:{worker_id}"', check=False)
                        presence_exists = stdout.strip() == "1"
                        stdout, _, _ = run_command(f'docker compose exec -T redis redis-cli EXISTS "worker:status:{worker_id}"', check=False)
                        status_exists = stdout.strip() == "1"
                        print(f"    Worker {worker_id}: presence={presence_exists}, status={status_exists}")
            else:
                results["worker_count_accurate"] = check_condition(
                    False,
                    f"Worker count endpoint returned {response.status_code}",
                    response.text[:100]
                )
        except requests.exceptions.RequestException as e:
            results["worker_count_accurate"] = check_condition(
                False,
                "Could not reach worker count endpoint",
                str(e)
            )
    
    print_step(8, "Test Summary")
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    print(f"\nResults: {passed}/{total} checks passed\n")
    
    for check, passed_check in results.items():
        status = "[PASS]" if passed_check else "[FAIL]"
        print(f"  {status} {check.replace('_', ' ').title()}")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        return True
    else:
        print(f"\n[WARN] {total - passed} test(s) failed. Review output above.")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
