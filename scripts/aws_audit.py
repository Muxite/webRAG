"""
AWS audit utilities - reusable OOP module for tracking changes.
"""
import json
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Dict, List
from pathlib import Path
import boto3
from botocore.exceptions import ClientError


class AWSAuditor:
    """Audits AWS changes and correlates with git."""
    
    def __init__(self, region: str, cluster: str, account_id: str = ""):
        self.region = region
        self.cluster = cluster
        self.account_id = account_id
        self.ecs = boto3.client("ecs", region_name=region)
        self.ecr = boto3.client("ecr", region_name=region)
        self.cloudtrail = boto3.client("cloudtrail", region_name=region)
    
    def get_git_commits(self, since: datetime, days: int = 7) -> List[Dict]:
        """Get git commits since time."""
        since_str = (since - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        cmd = ["git", "log", "--since", since_str, "--format=json", "--all", "--no-merges"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=Path.cwd())
            commits = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    try:
                        commits.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return commits
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []
    
    def get_cloudtrail_events(self, start: datetime, end: datetime, max_events: int = 50) -> List[Dict]:
        """Get CloudTrail events with timeout."""
        events = []
        try:
            response = self.cloudtrail.lookup_events(
                StartTime=start, EndTime=end, MaxResults=min(max_events, 50)
            )
            for event in response.get("Events", []):
                events.append({
                    "time": event.get("EventTime"),
                    "name": event.get("EventName"),
                    "source": event.get("EventSource", "").split(".")[0],
                    "user": event.get("Username", "N/A")
                })
        except (ClientError, Exception):
            pass
        return sorted(events, key=lambda x: x.get("time", datetime.min.replace(tzinfo=timezone.utc)))[:max_events]
    
    def get_task_def_revisions(self, family: str, start: datetime, end: datetime, max_rev: int = 10) -> List[Dict]:
        """Get task definition revisions in time window."""
        try:
            response = self.ecs.list_task_definitions(familyPrefix=family, sort="DESC", maxResults=max_rev)
            revisions = []
            for arn in response.get("taskDefinitionArns", [])[:max_rev]:
                try:
                    td = self.ecs.describe_task_definition(taskDefinition=arn).get("taskDefinition", {})
                    reg_time = td.get("registeredAt")
                    if reg_time:
                        if isinstance(reg_time, str):
                            reg_time = datetime.fromisoformat(reg_time.replace("Z", "+00:00"))
                        if start <= reg_time <= end:
                            revisions.append({
                                "revision": td.get("revision"),
                                "time": reg_time.isoformat(),
                                "cpu": td.get("cpu"),
                                "memory": td.get("memory"),
                                "containers": len(td.get("containerDefinitions", [])),
                                "container_names": [c.get("name") for c in td.get("containerDefinitions", [])]
                            })
                except ClientError:
                    continue
            return sorted(revisions, key=lambda x: x["time"], reverse=True)
        except ClientError:
            return []
    
    def get_ecs_events(self, service: str, start: datetime, end: datetime, max_events: int = 10) -> List[Dict]:
        """Get ECS service events."""
        try:
            response = self.ecs.describe_services(cluster=self.cluster, services=[service])
            if not response.get("services"):
                return []
            events = []
            for event in response["services"][0].get("events", [])[:max_events]:
                t_str = event.get("createdAt")
                if t_str:
                    try:
                        t = datetime.fromisoformat(t_str.replace("Z", "+00:00"))
                        if start <= t <= end:
                            events.append({
                                "time": t.isoformat(),
                                "message": event.get("message", "")[:200]
                            })
                    except (ValueError, TypeError):
                        continue
            return sorted(events, key=lambda x: x["time"], reverse=True)
        except ClientError:
            return []
    
    def get_ecr_pushes(self, repo: str, start: datetime, end: datetime, max_images: int = 5) -> List[Dict]:
        """Get ECR image pushes."""
        repo_name = f"euglena/{repo}" if not repo.startswith("euglena/") else repo
        try:
            response = self.ecr.describe_images(repositoryName=repo_name, maxResults=max_images)
            images = []
            for img in response.get("imageDetails", [])[:max_images]:
                pushed = img.get("imagePushedAt")
                if pushed:
                    if isinstance(pushed, str):
                        pushed = datetime.fromisoformat(pushed.replace("Z", "+00:00"))
                    if start <= pushed <= end:
                        images.append({
                            "time": pushed.isoformat(),
                            "tags": img.get("imageTags", [])[:3],
                            "digest": img.get("imageDigest", "")[:16]
                        })
            return sorted(images, key=lambda x: x["time"], reverse=True)
        except ClientError:
            return []
    
    def audit(self, target_time: datetime, days: int = 10) -> Dict:
        """Run full audit."""
        start = target_time - timedelta(days=days)
        end = datetime.now(timezone.utc) + timedelta(days=1)
        
        commits = self.get_git_commits(target_time, days)
        cloudtrail = self.get_cloudtrail_events(start, end)
        ecs_events = {}
        for svc in ["euglena-gateway", "euglena-agent", "euglena"]:
            evts = self.get_ecs_events(svc, start, end)
            if evts:
                ecs_events[svc] = evts
        
        task_defs = {}
        for family in ["euglena-gateway", "euglena-agent", "euglena"]:
            revs = self.get_task_def_revisions(family, start, end)
            if revs:
                task_defs[family] = revs
        
        ecr_pushes = {}
        for repo in ["gateway", "agent", "metrics"]:
            imgs = self.get_ecr_pushes(repo, start, end)
            if imgs:
                ecr_pushes[repo] = imgs
        
        return {
            "target_time": target_time.isoformat(),
            "window": {"start": start.isoformat(), "end": end.isoformat()},
            "git_commits": len(commits),
            "cloudtrail_events": len(cloudtrail),
            "ecs_events": {k: len(v) for k, v in ecs_events.items()},
            "task_definitions": {k: len(v) for k, v in task_defs.items()},
            "ecr_pushes": {k: len(v) for k, v in ecr_pushes.items()},
            "data": {
                "commits": commits,
                "cloudtrail": cloudtrail,
                "ecs_events": ecs_events,
                "task_defs": task_defs,
                "ecr_pushes": ecr_pushes
            }
        }
