#!/bin/bash
# AWS CLI commands to fix autoscaling EventBridge rule
# This enables the rule and sets it to trigger every 60 seconds

RULE_NAME="euglena-autoscale"
LAMBDA_NAME="euglena-autoscaling"

echo "=== Fixing EventBridge Rule ==="
echo "Rule name: $RULE_NAME"
echo "Lambda name: $LAMBDA_NAME"
echo ""

# Get Lambda ARN
LAMBDA_ARN=$(aws lambda get-function --function-name "$LAMBDA_NAME" --query 'Configuration.FunctionArn' --output text)

if [ -z "$LAMBDA_ARN" ]; then
    echo "ERROR: Lambda function '$LAMBDA_NAME' not found!"
    exit 1
fi

echo "Lambda ARN: $LAMBDA_ARN"
echo ""

# Update or create the rule
echo "Updating EventBridge rule..."
aws events put-rule \
  --name "$RULE_NAME" \
  --schedule-expression "rate(1 minute)" \
  --state ENABLED \
  --description "Triggers euglena autoscaling Lambda function every 60 seconds"

if [ $? -eq 0 ]; then
    echo "✓ Rule updated successfully"
else
    echo "✗ Failed to update rule"
    exit 1
fi

# Add or update the Lambda target
echo ""
echo "Configuring Lambda target..."
aws events put-targets \
  --rule "$RULE_NAME" \
  --targets "Id=1,Arn=$LAMBDA_ARN"

if [ $? -eq 0 ]; then
    echo "✓ Target configured successfully"
else
    echo "✗ Failed to configure target"
    exit 1
fi

# Verify
echo ""
echo "=== Verification ==="
aws events describe-rule --name "$RULE_NAME" --query '[State,ScheduleExpression]' --output table

echo ""
echo "✓ Autoscaling rule is now ENABLED and will trigger every 60 seconds"
