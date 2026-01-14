#!/usr/bin/env python3
"""
Fix EventBridge autoscaling rule configuration.

Enables the rule and sets the correct schedule expression.
"""

import boto3
import sys
import argparse
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
    """Fix autoscaling rule."""
    parser = argparse.ArgumentParser(description='Fix EventBridge autoscaling rule')
    parser.add_argument('--schedule', type=str, default='1 minute',
                       help='Schedule rate (e.g., "1 minute", "2 minutes", "60 seconds")')
    parser.add_argument('--rule-name', type=str, help='EventBridge rule name (overrides config)')
    parser.add_argument('--lambda-name', type=str, help='Lambda function name (overrides config)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    config = load_config()
    
    rule_name = args.rule_name or config.get('EVENTBRIDGE_RULE_NAME', 'euglena-autoscaling-trigger')
    lambda_name = args.lambda_name or config.get('LAMBDA_FUNCTION_NAME', 'euglena-autoscaling')
    
    schedule_expr = f"rate({args.schedule})"
    
    print(f"Fixing autoscaling rule...")
    print(f"Rule name: {rule_name}")
    print(f"Lambda name: {lambda_name}")
    print(f"Schedule: {schedule_expr}")
    print()
    
    eventbridge = boto3.client('events')
    lambda_client = boto3.client('lambda')
    
    try:
        if args.dry_run:
            print("[DRY RUN] Would update rule:")
            print(f"  Name: {rule_name}")
            print(f"  Schedule: {schedule_expr}")
            print(f"  State: ENABLED")
            return 0
        
        try:
            rule = eventbridge.describe_rule(Name=rule_name)
            print(f"Found existing rule: {rule_name}")
            print(f"  Current state: {rule.get('State')}")
            print(f"  Current schedule: {rule.get('ScheduleExpression')}")
        except eventbridge.exceptions.ResourceNotFoundException:
            print(f"Rule '{rule_name}' not found. Creating new rule...")
        
        print(f"\nUpdating rule...")
        eventbridge.put_rule(
            Name=rule_name,
            ScheduleExpression=schedule_expr,
            State='ENABLED',
            Description='Triggers euglena autoscaling Lambda function'
        )
        print(f"✓ Rule updated successfully")
        
        try:
            targets = eventbridge.list_targets_by_rule(Rule=rule_name)
            existing_targets = {t['Id']: t for t in targets.get('Targets', [])}
            
            lambda_arn = lambda_client.get_function(FunctionName=lambda_name)['Configuration']['FunctionArn']
            target_id = 'autoscaling-lambda-target'
            
            if target_id not in existing_targets:
                print(f"\nAdding Lambda target...")
                eventbridge.put_targets(
                    Rule=rule_name,
                    Targets=[{
                        'Id': target_id,
                        'Arn': lambda_arn
                    }]
                )
                print(f"✓ Target added successfully")
            else:
                existing_arn = existing_targets[target_id]['Arn']
                if existing_arn != lambda_arn:
                    print(f"\nUpdating Lambda target...")
                    eventbridge.remove_targets(Rule=rule_name, Ids=[target_id])
                    eventbridge.put_targets(
                        Rule=rule_name,
                        Targets=[{
                            'Id': target_id,
                            'Arn': lambda_arn
                        }]
                    )
                    print(f"✓ Target updated successfully")
                else:
                    print(f"\n✓ Target already configured correctly")
        except lambda_client.exceptions.ResourceNotFoundException:
            print(f"\n⚠️  Lambda '{lambda_name}' not found. Rule updated but target not configured.")
            print(f"   Configure the target manually in AWS Console or run this script again after creating the Lambda.")
        
        print()
        print("=== Verification ===")
        rule = eventbridge.describe_rule(Name=rule_name)
        print(f"Rule state: {rule.get('State')}")
        print(f"Schedule: {rule.get('ScheduleExpression')}")
        
        targets = eventbridge.list_targets_by_rule(Rule=rule_name)
        print(f"Targets: {len(targets.get('Targets', []))}")
        
        if rule.get('State') == 'ENABLED':
            print("\n✓ Rule is now ENABLED and should trigger every " + args.schedule)
        else:
            print("\n⚠️  Rule state is not ENABLED!")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
