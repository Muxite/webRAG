#!/usr/bin/env python3
"""
Diagnose and fix EventBridge autoscaling rule issues.

Checks:
- EventBridge rule state (enabled/disabled)
- Schedule expression
- Lambda target configuration
- Lambda function state
"""

import boto3
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

def load_config():
    """Load AWS configuration from aws.env."""
    project_root = Path(__file__).parent.parent
    aws_env_path = project_root / "services" / "aws.env"
    
    config = {}
    if aws_env_path.exists():
        from dotenv import dotenv_values
        config = dotenv_values(str(aws_env_path))
    
    return config

def main():
    """Check autoscaling configuration."""
    config = load_config()
    
    rule_name = config.get('EVENTBRIDGE_RULE_NAME', 'euglena-autoscaling-trigger')
    lambda_name = config.get('LAMBDA_FUNCTION_NAME', 'euglena-autoscaling')
    
    print(f"Checking autoscaling configuration...")
    print(f"Rule name: {rule_name}")
    print(f"Lambda name: {lambda_name}")
    print()
    
    eventbridge = boto3.client('events')
    lambda_client = boto3.client('lambda')
    logs = boto3.client('logs')
    
    try:
        print("=== EventBridge Rule ===")
        try:
            rule = eventbridge.describe_rule(Name=rule_name)
            print(f"✓ Rule found: {rule_name}")
            print(f"  State: {rule.get('State', 'UNKNOWN')}")
            print(f"  Schedule: {rule.get('ScheduleExpression', 'N/A')}")
            print(f"  Description: {rule.get('Description', 'N/A')}")
            
            if rule.get('State') != 'ENABLED':
                print(f"  ⚠️  WARNING: Rule is {rule.get('State')} - it will not trigger!")
            
            targets = eventbridge.list_targets_by_rule(Rule=rule_name)
            print(f"  Targets: {len(targets.get('Targets', []))}")
            for target in targets.get('Targets', []):
                print(f"    - {target.get('Id')}: {target.get('Arn', '').split(':')[-1]}")
                if target.get('Arn', '').split(':')[-1] != lambda_name:
                    print(f"      ⚠️  WARNING: Target doesn't match expected Lambda name!")
        except eventbridge.exceptions.ResourceNotFoundException:
            print(f"✗ Rule '{rule_name}' not found!")
            print("  Create it with:")
            print(f"    aws events put-rule --name {rule_name} --schedule-expression 'rate(1 minute)' --state ENABLED")
            return 1
        
        print()
        print("=== Lambda Function ===")
        try:
            func = lambda_client.get_function(FunctionName=lambda_name)
            config_info = func['Configuration']
            print(f"✓ Lambda found: {lambda_name}")
            print(f"  State: {config_info.get('State', 'UNKNOWN')}")
            print(f"  Last modified: {config_info.get('LastModified', 'N/A')}")
            print(f"  Runtime: {config_info.get('Runtime', 'N/A')}")
            print(f"  Timeout: {config_info.get('Timeout', 'N/A')}s")
            print(f"  Memory: {config_info.get('MemorySize', 'N/A')}MB")
            
            if config_info.get('State') != 'Active':
                print(f"  ⚠️  WARNING: Lambda is not in Active state!")
        except lambda_client.exceptions.ResourceNotFoundException:
            print(f"✗ Lambda '{lambda_name}' not found!")
            return 1
        
        print()
        print("=== Recent Invocations ===")
        log_group = f"/aws/lambda/{lambda_name}"
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=3)
            
            response = logs.filter_log_events(
                LogGroupName=log_group,
                StartTime=int(start_time.timestamp() * 1000),
                EndTime=int(end_time.timestamp() * 1000),
                FilterPattern='START'
            )
            
            events = response.get('events', [])
            print(f"Found {len(events)} START events in last 3 hours")
            
            if len(events) < 10:
                print("  ⚠️  WARNING: Very few invocations detected!")
                print("  Expected ~180 invocations for 60-second schedule")
                print("  Expected ~90 invocations for 2-minute schedule")
            
            if events:
                print("\n  Recent invocations:")
                for event in events[-5:]:
                    timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
                    print(f"    {timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        except logs.exceptions.ResourceNotFoundException:
            print(f"⚠️  Log group '{log_group}' not found (Lambda may not have been invoked yet)")
        
        print()
        print("=== Recommendations ===")
        if rule.get('State') != 'ENABLED':
            print("1. Enable the EventBridge rule:")
            print(f"   aws events enable-rule --name {rule_name}")
        
        schedule = rule.get('ScheduleExpression', '')
        if 'rate(1 minute)' not in schedule and 'rate(60 seconds)' not in schedule:
            print("2. Verify schedule expression matches your intent:")
            print(f"   Current: {schedule}")
            print("   For 60 seconds: rate(1 minute) or rate(60 seconds)")
            print("   For 2 minutes: rate(2 minutes)")
        
        print()
        print("=== Fix Command ===")
        print("To enable the rule and set to 60 seconds:")
        print(f"  aws events put-rule --name {rule_name} --schedule-expression 'rate(1 minute)' --state ENABLED")
        print()
        print("To enable the rule and set to 2 minutes:")
        print(f"  aws events put-rule --name {rule_name} --schedule-expression 'rate(2 minutes)' --state ENABLED")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
