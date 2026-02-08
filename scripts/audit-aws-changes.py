"""
Audit AWS changes and correlate with git commits.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

try:
    from scripts.deploy_common import load_aws_config
    from scripts.aws_audit import AWSAuditor
except ImportError:
    from deploy_common import load_aws_config
    from aws_audit import AWSAuditor


def parse_git_time(time_str: str) -> datetime:
    """Parse git time string."""
    try:
        return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S %z")
    except ValueError:
        return datetime.fromisoformat(time_str.replace("Z", "+00:00"))


def analyze_changes(audit_data: Dict) -> Dict:
    """Analyze audit data for issues."""
    issues = []
    data = audit_data["data"]
    
    for family, revs in data["task_defs"].items():
        for rev in revs:
            reg_time = datetime.fromisoformat(rev["time"].replace("Z", "+00:00"))
            nearby = [c for c in data["commits"] 
                     if abs((parse_git_time(c.get("author", {}).get("date", "")) - reg_time).total_seconds()) < 3600]
            if not nearby:
                issues.append({
                    "type": "orphaned_task_definition",
                    "family": family,
                    "revision": rev["revision"],
                    "time": rev["time"],
                    "note": "No nearby git commit"
                })
    
    for family, revs in data["task_defs"].items():
        prev_cnt = None
        for rev in sorted(revs, key=lambda x: x["time"]):
            if prev_cnt and rev["containers"] != prev_cnt:
                issues.append({
                    "type": "container_count_change",
                    "family": family,
                    "revision": rev["revision"],
                    "time": rev["time"],
                    "note": f"Containers: {prev_cnt} -> {rev['containers']}"
                })
            prev_cnt = rev["containers"]
    
    return {"issues": issues, "summary": audit_data}


def parse_args():
    """
    Parse CLI arguments.

    :returns: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Audit AWS changes and correlate with git commits")
    parser.add_argument("--target-time", type=str, 
                       default="2026-02-01 15:56:28 -0800",
                       help="Target time to investigate (format: YYYY-MM-DD HH:MM:SS TZ)")
    parser.add_argument("--days", type=int, default=10,
                       help="Number of days to look back (default: 10 to capture working state)")
    parser.add_argument("--hours-window", type=int, default=24,
                       help="Hours window around target time for git commits")
    parser.add_argument("--output", type=str,
                       help="Output JSON file path")
    parser.add_argument("--region", type=str,
                       help="AWS region (overrides aws.env)")
    parser.add_argument("--cluster", type=str,
                       help="ECS cluster name (overrides aws.env)")
    
    return parser.parse_args()

def main():
    """Main audit function."""
    import argparse
    
    args = parse_args()
    
    try:
        target_time = parse_git_time(args.target_time)
    except ValueError:
        print(f"Error: Invalid time format: {args.target_time}", file=sys.stderr)
        print("Expected format: YYYY-MM-DD HH:MM:SS TZ (e.g., '2026-02-01 15:56:28 -0800')", file=sys.stderr)
        sys.exit(1)
    
    services_dir = Path.cwd()
    if (services_dir / "services").exists():
        services_dir = services_dir / "services"
    
    aws_config = load_aws_config(services_dir)
    region = args.region or aws_config.get("AWS_REGION", "us-east-2")
    cluster = args.cluster or aws_config.get("ECS_CLUSTER_NAME", "euglena-cluster")
    account_id = aws_config.get("AWS_ACCOUNT_ID", "")
    
    auditor = AWSAuditor(region, cluster, account_id)
    audit_data = auditor.audit(target_time, args.days)
    analysis = analyze_changes(audit_data)
    
    print(f"Audit: {target_time.isoformat()} | {args.days}d | {region}/{cluster}")
    print(f"Git: {audit_data['git_commits']} commits")
    print(f"CloudTrail: {audit_data['cloudtrail_events']} events")
    print(f"ECS Events: {sum(audit_data['ecs_events'].values())} ({audit_data['ecs_events']})")
    print(f"TaskDefs: {sum(audit_data['task_definitions'].values())} revs")
    for f, revs in audit_data["data"]["task_defs"].items():
        if revs:
            rev_strs = [f"r{r['revision']}({r['cpu']}/{r['memory']}/{r['containers']})" for r in revs[:3]]
            print(f"  {f}: {rev_strs}")
    print(f"ECR: {sum(audit_data['ecr_pushes'].values())} pushes")
    
    issues = analysis["issues"]
    print(f"Issues: {len(issues)}")
    for issue in issues[:8]:
        print(f"  [{issue['type']}] {issue['time'][:16]}: {issue['note'][:50]}")
    
    results = {
        "target_time": target_time.isoformat(),
        "window": audit_data["window"],
        "summary": audit_data,
        "analysis": analysis
    }
    
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\n=== Summary JSON ===")
        print(json.dumps({
            "summary": analysis["summary"],
            "issues_count": len(analysis["potential_issues"]),
            "timeline_events": len(analysis["timeline"])
        }, indent=2, default=str))


if __name__ == "__main__":
    main()
