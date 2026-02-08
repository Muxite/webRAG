"""
Analyze audit results to identify breaking point.
"""
import json
import sys
from datetime import datetime
from pathlib import Path


def parse_time(time_str):
    """Parse ISO time string."""
    try:
        if "T" in time_str:
            return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        return datetime.fromisoformat(time_str)
    except:
        return None


def analyze_audit(json_path):
    """Analyze audit results."""
    with open(json_path) as f:
        data = json.load(f)
    
    target_time = parse_time(data["target_time"])
    
    print("=" * 80)
    print("BREAKING POINT ANALYSIS")
    print("=" * 80)
    print(f"\nTarget (working) time: {target_time}")
    window = data.get("window") or data.get("time_window", {})
    if window:
        print(f"Time window: {window.get('start', 'N/A')} to {window.get('end', 'N/A')}")
    
    # Analyze task definitions
    print("\n" + "=" * 80)
    print("TASK DEFINITION TIMELINE")
    print("=" * 80)
    
    task_defs = data.get("summary", {}).get("data", {}).get("task_defs", {})
    gateway_tds = task_defs.get("euglena-gateway", [])
    agent_tds = task_defs.get("euglena-agent", [])
    single_tds = task_defs.get("euglena", [])
    
    # Find "good" state (around target time)
    print("\n--- GOOD STATE (around target time) ---")
    good_gateway = None
    good_agent = None
    
    for td in gateway_tds:
        reg_time = parse_time(td["registeredAt"])
        if reg_time and abs((reg_time - target_time).total_seconds()) < 86400:  # Within 24h
            good_gateway = td
            print(f"Gateway r{td['revision']}: {td['registeredAt']}")
            print(f"  CPU: {td['cpu']}, Memory: {td['memory']}, Containers: {td['containerCount']}")
            break
    
    for td in agent_tds:
        reg_time = parse_time(td["registeredAt"])
        if reg_time and abs((reg_time - target_time).total_seconds()) < 86400:
            good_agent = td
            print(f"Agent r{td['revision']}: {td['registeredAt']}")
            print(f"  CPU: {td['cpu']}, Memory: {td['memory']}, Containers: {td['containerCount']}")
            break
    
    # Find breaking point (first failure after good state)
    print("\n--- BREAKING POINT ANALYSIS ---")
    
    # Gateway failures
    gateway_failures = []
    for event in data["ecs_service_events"]["euglena-gateway"]:
        msg = event["message"]
        if "failed" in msg.lower() or "unhealthy" in msg.lower() or "deployment failed" in msg.lower():
            gateway_failures.append(event)
    
    if gateway_failures:
        first_failure = gateway_failures[-1]  # Most recent
        failure_time = parse_time(first_failure["createdAt"])
        print(f"\nFirst gateway failure: {first_failure['createdAt']}")
        print(f"  Message: {first_failure['message']}")
        
        # Find task def in use at failure time
        active_td = None
        for td in sorted(gateway_tds, key=lambda x: parse_time(x["registeredAt"]) or datetime.min, reverse=True):
            reg_time = parse_time(td["registeredAt"])
            if reg_time and reg_time <= failure_time:
                active_td = td
                break
        
        if active_td:
            print(f"\nTask definition in use at failure: r{active_td['revision']}")
            print(f"  Registered: {active_td['registeredAt']}")
            print(f"  CPU: {active_td['cpu']}, Memory: {active_td['memory']}, Containers: {active_td['containerCount']}")
            
            # Compare with good state
            if good_gateway:
                print(f"\nComparison with good state (r{good_gateway['revision']}):")
                changes = []
                if active_td['cpu'] != good_gateway['cpu']:
                    changes.append(f"CPU: {good_gateway['cpu']} -> {active_td['cpu']}")
                if active_td['memory'] != good_gateway['memory']:
                    changes.append(f"Memory: {good_gateway['memory']} -> {active_td['memory']}")
                if active_td['containerCount'] != good_gateway['containerCount']:
                    changes.append(f"Containers: {good_gateway['containerCount']} -> {active_td['containerCount']}")
                
                if changes:
                    print(f"  CHANGES: {', '.join(changes)}")
                else:
                    print(f"  No resource changes (likely code/image issue)")
    
    # Analyze timeline of changes
    print("\n--- TIMELINE OF CHANGES ---")
    print("\nGateway task definitions:")
    for td in sorted(gateway_tds, key=lambda x: parse_time(x["registeredAt"]) or datetime.min):
        reg_time = parse_time(td["registeredAt"])
        if reg_time:
            time_diff = (reg_time - target_time).total_seconds() / 3600
            marker = " <-- GOOD" if td == good_gateway else ""
            marker += " <-- FAILURE" if td == active_td else ""
            print(f"  r{td['revision']}: {td['registeredAt'][:16]} ({time_diff:+.1f}h) "
                  f"CPU:{td['cpu']} MEM:{td['memory']} CNT:{td['containerCount']}{marker}")
    
    # Find suspicious changes
    print("\n--- SUSPICIOUS CHANGES ---")
    
    # Container count changes
    prev_count = None
    for td in sorted(gateway_tds, key=lambda x: parse_time(x["registeredAt"]) or datetime.min):
        if prev_count is not None and td['containerCount'] != prev_count:
            reg_time = parse_time(td["registeredAt"])
            time_diff = (reg_time - target_time).total_seconds() / 3600 if reg_time else 0
            print(f"  Container count changed: r{td['revision']} ({time_diff:+.1f}h) "
                  f"{prev_count} -> {td['containerCount']} containers")
        prev_count = td['containerCount']
    
    # Resource changes
    prev_cpu = None
    prev_mem = None
    for td in sorted(gateway_tds, key=lambda x: parse_time(x["registeredAt"]) or datetime.min):
        if prev_cpu is not None and (td['cpu'] != prev_cpu or td['memory'] != prev_mem):
            reg_time = parse_time(td["registeredAt"])
            time_diff = (reg_time - target_time).total_seconds() / 3600 if reg_time else 0
            print(f"  Resource change: r{td['revision']} ({time_diff:+.1f}h) "
                  f"CPU:{prev_cpu}->{td['cpu']} MEM:{prev_mem}->{td['memory']}")
        prev_cpu = td['cpu']
        prev_mem = td['memory']
    
    # ECR push analysis
    print("\n--- ECR IMAGE PUSHES ---")
    for repo, images in data["ecr_image_pushes"].items():
        for img in images:
            push_time = parse_time(img["imagePushedAt"])
            if push_time:
                time_diff = (push_time - target_time).total_seconds() / 3600
                tags = img.get("imageTags", ["<untagged>"])[:3]
                print(f"  {repo}: {img['imagePushedAt'][:16]} ({time_diff:+.1f}h) tags:{tags}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if good_gateway and active_td:
        print(f"\nGood state: Gateway r{good_gateway['revision']} ({good_gateway['registeredAt'][:16]})")
        print(f"Failure state: Gateway r{active_td['revision']} ({active_td['registeredAt'][:16]})")
        
        if good_gateway['revision'] != active_td['revision']:
            print(f"\n[!] Task definition changed from r{good_gateway['revision']} to r{active_td['revision']}")
            print(f"   This suggests the failure is due to a task definition change, not code.")
        else:
            print(f"\n[!] Same task definition (r{good_gateway['revision']}) but failing now.")
            print(f"   This suggests AWS infrastructure or external dependency changed.")
    
    # Key findings
    print("\n--- KEY FINDINGS ---")
    
    # Check for container count drop
    if good_gateway:
        for td in gateway_tds:
            if td['containerCount'] < good_gateway['containerCount']:
                reg_time = parse_time(td["registeredAt"])
                if reg_time and reg_time > target_time:
                    print(f"[!] Container count dropped in r{td['revision']}: "
                          f"{good_gateway['containerCount']} -> {td['containerCount']}")
                    print(f"   This might indicate missing metrics container or other container.")
    
    # Check for resource changes
    if good_gateway:
        for td in gateway_tds:
            reg_time = parse_time(td["registeredAt"])
            if reg_time and reg_time > target_time:
                if td['cpu'] != good_gateway['cpu'] or td['memory'] != good_gateway['memory']:
                    print(f"[!] Resource change in r{td['revision']}: "
                          f"CPU {good_gateway['cpu']}->{td['cpu']}, "
                          f"MEM {good_gateway['memory']}->{td['memory']}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    json_path = sys.argv[1] if len(sys.argv) > 1 else "audit_results.json"
    if not Path(json_path).exists():
        print(f"Error: {json_path} not found", file=sys.stderr)
        sys.exit(1)
    analyze_audit(json_path)
