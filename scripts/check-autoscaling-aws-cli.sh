#!/bin/bash
# AWS CLI commands to diagnose autoscaling EventBridge rule
# Run these commands to check the rule status

RULE_NAME="euglena-autoscale"
LAMBDA_NAME="euglena-autoscaling"

echo "=== Checking EventBridge Rule ==="
aws events describe-rule --name "$RULE_NAME" 2>&1

echo ""
echo "=== Checking Rule Targets ==="
aws events list-targets-by-rule --rule "$RULE_NAME" 2>&1

echo ""
echo "=== Checking Lambda Function ==="
aws lambda get-function --function-name "$LAMBDA_NAME" --query 'Configuration.[State,LastModified,Timeout,MemorySize]' 2>&1

echo ""
echo "=== Recent Lambda Invocations (last 3 hours) ==="
LOG_GROUP="/aws/lambda/$LAMBDA_NAME"
START_TIME=$(date -u -d '3 hours ago' +%s)000
END_TIME=$(date -u +%s)000

aws logs filter-log-events \
  --log-group-name "$LOG_GROUP" \
  --start-time "$START_TIME" \
  --end-time "$END_TIME" \
  --filter-pattern "START" \
  --query 'events[*].timestamp' \
  --output text 2>&1 | wc -l

echo "invocations found"
