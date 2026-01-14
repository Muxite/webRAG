# Quick diagnostic script to check Lambda queue depth readings

$LAMBDA_NAME = "euglena-autoscale"
$LOG_GROUP = "/aws/lambda/$LAMBDA_NAME"

Write-Host "=== Lambda Queue Depth Check ===" -ForegroundColor Cyan
Write-Host ""

# Check recent invocations
Write-Host "Recent invocations (last 5 minutes):" -ForegroundColor Yellow
$invocations = aws logs filter-log-events `
  --log-group-name $LOG_GROUP `
  --start-time ([DateTimeOffset]::UtcNow.AddMinutes(-5).ToUnixTimeMilliseconds()) `
  --end-time ([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()) `
  --filter-pattern "START" `
  --query 'length(events)' `
  --output text
Write-Host "  Found: $invocations invocations" -ForegroundColor $(if ($invocations -gt 0) { "Green" } else { "Red" })
Write-Host ""

# Check queue depth readings
Write-Host "Queue depth readings (last 10 minutes):" -ForegroundColor Yellow
$readings = aws logs filter-log-events `
  --log-group-name $LOG_GROUP `
  --start-time ([DateTimeOffset]::UtcNow.AddMinutes(-10).ToUnixTimeMilliseconds()) `
  --end-time ([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()) `
  --filter-pattern "Queue depth" `
  --query 'events[*].message' `
  --output text

if ($readings) {
    $readings -split "`n" | ForEach-Object {
        if ($_ -match "Queue depth: (\d+)") {
            Write-Host "  $_" -ForegroundColor Green
        } else {
            Write-Host "  $_" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "  No queue depth readings found" -ForegroundColor Red
}
Write-Host ""

# Check scaling actions
Write-Host "Scaling actions (last 10 minutes):" -ForegroundColor Yellow
$scaling = aws logs filter-log-events `
  --log-group-name $LOG_GROUP `
  --start-time ([DateTimeOffset]::UtcNow.AddMinutes(-10).ToUnixTimeMilliseconds()) `
  --end-time ([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()) `
  --filter-pattern "Scaling" `
  --query 'events[*].message' `
  --output text

if ($scaling) {
    $scaling -split "`n" | ForEach-Object {
        Write-Host "  $_" -ForegroundColor Cyan
    }
} else {
    Write-Host "  No scaling actions" -ForegroundColor Gray
}
Write-Host ""

# Check for errors
Write-Host "Errors (last 10 minutes):" -ForegroundColor Yellow
$errors = aws logs filter-log-events `
  --log-group-name $LOG_GROUP `
  --start-time ([DateTimeOffset]::UtcNow.AddMinutes(-10).ToUnixTimeMilliseconds()) `
  --end-time ([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()) `
  --filter-pattern "ERROR" `
  --query 'events[*].message' `
  --output text

if ($errors) {
    $errors -split "`n" | ForEach-Object {
        Write-Host "  $_" -ForegroundColor Red
    }
} else {
    Write-Host "  No errors found" -ForegroundColor Green
}
Write-Host ""

# Check CloudWatch metrics
Write-Host "CloudWatch QueueDepth metrics (last hour):" -ForegroundColor Yellow
$metrics = aws cloudwatch get-metric-statistics `
  --namespace "Euglena/RabbitMQ" `
  --metric-name "QueueDepth" `
  --dimensions Name=QueueName,Value=agent.mandates `
  --start-time (Get-Date).AddHours(-1).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss") `
  --end-time (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss") `
  --period 300 `
  --statistics Average `
  --query 'Datapoints[*].[Timestamp,Average]' `
  --output json | ConvertFrom-Json

if ($metrics -and $metrics.Count -gt 0) {
    Write-Host "  Found $($metrics.Count) datapoints:" -ForegroundColor Green
    $metrics | ForEach-Object {
        $timestamp = [DateTimeOffset]::Parse($_[0]).ToLocalTime().ToString("HH:mm:ss")
        $value = $_[1]
        Write-Host "    [$timestamp] Queue depth: $value" -ForegroundColor Green
    }
} else {
    Write-Host "  No metrics found in CloudWatch!" -ForegroundColor Red
    Write-Host "  This means queue depth metrics are not being published." -ForegroundColor Yellow
    Write-Host "  Check if your gateway/service is publishing metrics to:" -ForegroundColor Yellow
    Write-Host "    Namespace: Euglena/RabbitMQ" -ForegroundColor Yellow
    Write-Host "    Metric: QueueDepth" -ForegroundColor Yellow
    Write-Host "    Dimension: QueueName=agent.mandates" -ForegroundColor Yellow
}
Write-Host ""

Write-Host "=== Summary ===" -ForegroundColor Cyan
if ($invocations -gt 0 -and $readings) {
    Write-Host "✓ Lambda is running and reading queue depth" -ForegroundColor Green
} elseif ($invocations -gt 0 -and -not $readings) {
    Write-Host "⚠ Lambda is running but not finding queue depth metrics" -ForegroundColor Yellow
} else {
    Write-Host "✗ Lambda is not being invoked" -ForegroundColor Red
}
