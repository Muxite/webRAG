# Chroma Health Check Failure Modes

**Note**: Chroma is optional - the agent is designed to continue working even if Chroma health checks fail. Chroma may still be functional for operations even when marked unhealthy by ECS.

## Potential Failure Points

### 1. Network & Connectivity Issues

**Connection Timeouts:**
- Chroma might not be listening on port 8000 yet (startup delay)
- Port binding failures (port already in use)
- Network interface not ready
- DNS resolution failures (though using localhost should avoid this)

**Solution**: Added `--connect-timeout 5` to fail fast on connection issues

**Connection Refused:**
- Chroma process crashed or not started
- Wrong port configuration
- Firewall/security group blocking localhost connections

**Solution**: `curl -f` will fail on connection refused

### 2. HTTP Response Issues

**Non-200 Status Codes:**
- Chroma might return 500 (internal error) but still be "running"
- Chroma might return 503 (service unavailable) during initialization
- Chroma might return 404 if endpoint changed

**Solution**: `curl -f` fails on any HTTP error status (4xx, 5xx)

**Slow Response:**
- Chroma overloaded and slow to respond
- Model loading causing delays
- EFS I/O latency

**Solution**: Added `--max-time 15` to timeout after 15 seconds

### 3. EFS/Storage Issues

**EFS Mount Failures:**
- EFS mount target not available
- EFS permissions issues
- EFS performance throttling
- EFS file system in error state

**Impact**: Chroma might start but fail to persist data or load cached models

**Detection**: Health check might pass but Chroma operations fail

**EFS Space Exhaustion:**
- Running out of EFS storage space
- Chroma unable to write to `/chroma-data`

**Impact**: Chroma might respond to heartbeat but fail on write operations

### 4. Resource Constraints

**Memory Exhaustion:**
- Chroma OOM kills during model loading
- Insufficient memory for embedding operations
- Memory leaks causing gradual degradation

**CPU Throttling:**
- Chroma slow to respond under CPU constraints
- Health check times out due to CPU starvation

**Solution**: Increased timeout to 20 seconds, increased retries to 10

### 5. Chroma Service Issues

**Startup Delays:**
- Model download/initialization taking longer than expected
- Database initialization delays
- EFS mount taking time to become available

**Solution**: 300-second start period allows time for initialization

**Degraded State:**
- Chroma responding to heartbeat but not functional
- Database corruption
- Collection creation failures

**Detection**: Health check passes but actual operations fail

### 6. Health Check Command Issues

**curl Not Available:**
- curl not installed in Chroma container
- PATH issues preventing curl execution

**Solution**: Chroma container should have curl (verify in Dockerfile)

**Exit Code Propagation:**
- Shell might not properly propagate exit codes
- Health check might not fail on errors

**Solution**: Explicit `|| exit 1` ensures failure propagation

**Silent Failures:**
- curl might fail silently without proper error handling
- Output redirection might hide errors

**Solution**: Added `-s -S` flags: `-s` silences progress, `-S` shows errors

### 7. Configuration Issues

**Wrong Endpoint:**
- Chroma version change might move endpoint
- Endpoint might require authentication
- Endpoint might be disabled

**Solution**: Verify `/api/v1/heartbeat` is correct for Chroma version

**Environment Variables:**
- Model cache paths misconfigured
- Persistence directory not writable
- Missing required environment variables

**Impact**: Chroma might start but fail to function properly

### 8. Timing Issues

**Race Conditions:**
- Health check runs before Chroma is ready
- Health check runs during Chroma restart
- Health check runs during model download

**Solution**: 300-second start period + 10 retries with 60-second intervals

**Clock Skew:**
- Container clock out of sync
- Health check timing calculations off

**Impact**: Minimal, but could affect retry logic

### 9. Container Lifecycle Issues

**Container Restart Loop:**
- Chroma crashes immediately after start
- Health check passes but container exits
- Dependency failures causing restarts

**Detection**: Monitor container restart counts

**Task Definition Changes:**
- Health check configuration changes
- Environment variable changes
- Volume mount changes

**Impact**: Health check behavior changes unexpectedly

### 10. Integration Issues

**Agent-Chroma Communication:**
- Agent can't reach Chroma even if health check passes
- Network namespace issues
- Port mapping problems

**Detection**: Agent health check includes Chroma connectivity test

**Dependency Failures:**
- Chroma depends on other services
- Service discovery failures
- DNS resolution issues

## Improved Health Check Command

Current command:
```bash
curl -f -s -S --max-time 15 --connect-timeout 5 http://localhost:8000/api/v1/heartbeat > /dev/null 2>&1 || exit 1
```

**Flags Explained:**
- `-f`: Fail on HTTP error status codes (4xx, 5xx)
- `-s`: Silent mode (no progress bar)
- `-S`: Show errors even in silent mode
- `--max-time 15`: Maximum time for entire operation (15 seconds)
- `--connect-timeout 5`: Maximum time to establish connection (5 seconds)
- `> /dev/null 2>&1`: Redirect output to avoid log noise
- `|| exit 1`: Explicitly exit with error code on failure

## Health Check Configuration

Current settings:
- **Start Period**: 300 seconds (5 minutes grace period)
- **Interval**: 60 seconds (check every minute)
- **Retries**: 6 (allow 6 consecutive failures)
- **Timeout**: 20 seconds (ECS timeout for health check execution)

**Total Failure Window**: 6 retries × 60 seconds = 6 minutes before marking unhealthy (after 5-minute start period)

**Task Resources**: 1 vCPU (1024 CPU units), 2GB RAM (2048 MB)

**Health Check Command**: Simplified to `curl -f http://localhost:8000/api/v1/heartbeat || (echo 'Chroma health check failed' && exit 1)`
- Removed complex timeout flags to avoid premature failures
- Added error message for better logging
- Basic connectivity test focuses on endpoint availability

## Recommendations

1. **Monitor Chroma Logs**: Check for startup errors, model download issues, EFS mount problems
2. **Verify EFS Access**: Ensure EFS mount targets are available and accessible
3. **Check Resource Usage**: Monitor CPU/memory usage during health check failures
4. **Test Actual Operations**: Health check passing doesn't guarantee Chroma is functional
5. **Add Operational Health Check**: Consider adding a check that tests actual Chroma operations (not just heartbeat)
