# PowerShell script to create/fix autoscaling EventBridge rule
# This creates or enables the rule and sets it to trigger every 60 seconds

$RULE_NAME = "euglena-autoscale"
$LAMBDA_NAME = "euglena-autoscaling"

Write-Host "=== Creating/Fixing EventBridge Rule ===" -ForegroundColor Cyan
Write-Host "Rule name: $RULE_NAME"
Write-Host "Lambda name: $LAMBDA_NAME"
Write-Host ""

# Get Lambda ARN
Write-Host "Getting Lambda function ARN..." -ForegroundColor Yellow
$lambdaResponse = aws lambda get-function --function-name $LAMBDA_NAME --query 'Configuration.FunctionArn' --output text 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Lambda function '$LAMBDA_NAME' not found!" -ForegroundColor Red
    Write-Host $lambdaResponse
    Write-Host ""
    Write-Host "Please create the Lambda function first, or check the function name." -ForegroundColor Yellow
    exit 1
}

$LAMBDA_ARN = $lambdaResponse.Trim()
Write-Host "✓ Lambda ARN: $LAMBDA_ARN" -ForegroundColor Green
Write-Host ""

# Check if rule exists
Write-Host "Checking if rule exists..." -ForegroundColor Yellow
$ruleCheck = aws events describe-rule --name $RULE_NAME 2>&1
$ruleExists = $LASTEXITCODE -eq 0

if ($ruleExists) {
    Write-Host "✓ Rule exists, updating..." -ForegroundColor Green
} else {
    Write-Host "Rule does not exist, creating new rule..." -ForegroundColor Yellow
}

# Create or update the rule
Write-Host ""
Write-Host "Creating/updating EventBridge rule..." -ForegroundColor Yellow
aws events put-rule `
  --name $RULE_NAME `
  --schedule-expression "rate(1 minute)" `
  --state ENABLED `
  --description "Triggers euglena autoscaling Lambda function every 60 seconds"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Rule created/updated successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to create/update rule" -ForegroundColor Red
    exit 1
}

# Check existing targets
Write-Host ""
Write-Host "Checking existing targets..." -ForegroundColor Yellow
$existingTargets = aws events list-targets-by-rule --rule $RULE_NAME --query 'Targets[*].Id' --output text 2>&1

# Remove old targets if they exist
if ($LASTEXITCODE -eq 0 -and $existingTargets) {
    Write-Host "Removing old targets..." -ForegroundColor Yellow
    $targetIds = $existingTargets.Trim() -split '\s+'
    foreach ($id in $targetIds) {
        if ($id) {
            aws events remove-targets --rule $RULE_NAME --ids $id | Out-Null
        }
    }
}

# Add the Lambda target
Write-Host "Configuring Lambda target..." -ForegroundColor Yellow
aws events put-targets `
  --rule $RULE_NAME `
  --targets "Id=1,Arn=$LAMBDA_ARN"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Target configured successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to configure target" -ForegroundColor Red
    exit 1
}

# Verify
Write-Host ""
Write-Host "=== Verification ===" -ForegroundColor Cyan
$ruleInfo = aws events describe-rule --name $RULE_NAME --query '[Name,State,ScheduleExpression,Description]' --output table
Write-Host $ruleInfo

$targets = aws events list-targets-by-rule --rule $RULE_NAME --query 'Targets[*].[Id,Arn]' --output table
Write-Host ""
Write-Host "Targets:" -ForegroundColor Cyan
Write-Host $targets

Write-Host ""
Write-Host "✓ Autoscaling rule is now ENABLED and will trigger every 60 seconds" -ForegroundColor Green
Write-Host "  The Lambda function will be invoked approximately every minute." -ForegroundColor Gray
