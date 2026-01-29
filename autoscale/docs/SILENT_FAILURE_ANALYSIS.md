# Silent Service Failure Analysis

## Problem
Gateway service (bundle of gateway, chroma, redis, rabbitmq, metrics) mysteriously fails health checks and dies without logs, even after extensive exception handling was added.

## Potential Root Causes

### 1. **Out of Memory (OOM) Kills**
**Most Likely Cause**

- **Symptom**: Process dies instantly with no logs
- **Why**: Linux OOM killer sends SIGKILL (cannot be caught)
- **Detection**: Check `dmesg`, container exit code 137, Docker logs
- **Prevention**:
  - Set memory limits in docker-compose
  - Monitor memory usage
  - Add memory monitoring endpoint

### 2. **Health Check Timeout Causing Orchestration Kill**
**High Probability**

- **Symptom**: Service dies after health check failures
- **Why**: Health check hangs → orchestration kills container
- **Current Issue**: Health check has timeout but may still hang if:
  - Redis connection hangs (no timeout on `init_redis()`)
  - RabbitMQ check blocks
  - Exception in health check itself
- **Fix Needed**: 
  - Add overall health check timeout wrapper
  - Ensure all health checks have timeouts
  - Add circuit breaker to health checks

### 3. **Logging Handler Failure**
**Medium Probability**

- **Symptom**: Exceptions occur but logging fails, so nothing appears
- **Why**: 
  - Logging handler throws exception
  - Log buffer full
  - Disk full
  - Permission issues
- **Current Risk**: If `logger.error()` throws, exception is lost
- **Fix Needed**: 
  - Add fallback logging to stderr
  - Wrap all logging calls
  - Add logging health check

### 4. **Uvicorn Worker Crash**
**Medium Probability**

- **Symptom**: Process exits without FastAPI exception handlers firing
- **Why**: 
  - Uvicorn worker process crashes before exception reaches FastAPI
  - Background thread/process failure
  - Signal handling issues
- **Current Risk**: No signal handlers for SIGTERM/SIGINT
- **Fix Needed**:
  - Add signal handlers
  - Use uvicorn with multiple workers (for resilience)
  - Add process monitoring

### 5. **Lifespan Startup Failure**
**Medium Probability**

- **Symptom**: Service dies during startup, no logs
- **Why**: 
  - Exception in `lifespan()` startup
  - Exception handler not initialized yet
  - Logging not configured yet
- **Current Risk**: If `gateway_service.start()` throws before logging setup
- **Fix Needed**:
  - Initialize logging before lifespan
  - Wrap lifespan startup in try/except with stderr fallback
  - Add startup timeout

### 6. **Background Task Failures**
**Low-Medium Probability**

- **Symptom**: Service appears healthy but stops processing
- **Why**: 
  - Background asyncio task crashes
  - No exception handling in background tasks
  - Task cancellation not handled
- **Current Risk**: No background tasks visible in gateway, but agent has reconnect loops
- **Fix Needed**: 
  - Wrap all background tasks
  - Add task monitoring
  - Log task exceptions

### 7. **Exception Handler Exception**
**Low Probability**

- **Symptom**: Exception occurs, handler throws, nothing logged
- **Why**: Exception handler itself throws exception
- **Current Risk**: If `logger.error()` in handler throws
- **Fix Needed**: 
  - Add try/except in exception handler
  - Fallback to stderr
  - Add handler health check

### 8. **Database/Connection Pool Exhaustion**
**Low-Medium Probability**

- **Symptom**: Service hangs then dies
- **Why**: 
  - Redis connection pool exhausted
  - RabbitMQ connection limit reached
  - All connections in bad state
- **Current Risk**: No connection pool monitoring
- **Fix Needed**:
  - Add connection pool limits
  - Monitor connection counts
  - Add connection health checks

### 9. **Docker/K8s Health Check Failure**
**High Probability**

- **Symptom**: Container killed by orchestration
- **Why**: 
  - Health check endpoint returns non-200
  - Health check times out
  - Health check endpoint crashes
- **Current Risk**: Health check can hang or throw
- **Fix Needed**:
  - Add health check timeout wrapper
  - Ensure health check always returns
  - Add health check circuit breaker

### 10. **Python Interpreter Crash**
**Low Probability**

- **Symptom**: Process exits with no Python exception
- **Why**: 
  - Segmentation fault in C extension
  - Stack overflow
  - Interpreter bug
- **Detection**: Check for core dumps, exit codes
- **Fix Needed**: 
  - Enable core dumps
  - Add signal handlers
  - Monitor exit codes

## Recommended Fixes

### Immediate Actions

1. **Add Signal Handlers**
   ```python
   import signal
   import sys
   
   def signal_handler(sig, frame):
       logger.critical(f"Received signal {sig}, shutting down gracefully")
       # Cleanup
       sys.exit(0)
   
   signal.signal(signal.SIGTERM, signal_handler)
   signal.signal(signal.SIGINT, signal_handler)
   ```

2. **Add Health Check Timeout Wrapper**
   ```python
   @router.get("/health")
   async def health_check():
       try:
           return await asyncio.wait_for(_do_health_check(), timeout=4.0)
       except asyncio.TimeoutError:
           return {"status": "unhealthy", "reason": "health_check_timeout"}
       except Exception as e:
           # Log to stderr as fallback
           print(f"Health check error: {e}", file=sys.stderr)
           return {"status": "unhealthy", "reason": str(e)}
   ```

3. **Add Fallback Logging**
   ```python
   def safe_log(logger, level, message, *args, **kwargs):
       try:
           getattr(logger, level)(message, *args, **kwargs)
       except Exception:
           print(f"[FALLBACK {level.upper()}] {message}", file=sys.stderr)
   ```

4. **Add Memory Monitoring**
   ```python
   @router.get("/metrics/memory")
   async def memory_metrics():
       import psutil
       process = psutil.Process()
       return {
           "rss_mb": process.memory_info().rss / 1024 / 1024,
           "vms_mb": process.memory_info().vms / 1024 / 1024,
           "percent": process.memory_percent()
       }
   ```

5. **Wrap Lifespan Startup**
   ```python
   @asynccontextmanager
   async def lifespan(app: FastAPI):
       try:
           # Initialize logging first
           logger = setup_service_logger("Gateway", logging.INFO)
           log_startup_message(logger, "GATEWAY", "0.1.0")
           
           # Startup with timeout
           await asyncio.wait_for(
               app.state.gateway_service.start(),
               timeout=30.0
           )
           yield
       except Exception as e:
           # Fallback logging
           print(f"Lifespan startup failed: {e}", file=sys.stderr)
           import traceback
           traceback.print_exc(file=sys.stderr)
           raise
       finally:
           # Shutdown
   ```

6. **Add Process Monitoring**
   - Log process ID on startup
   - Log memory usage periodically
   - Log thread/async task counts
   - Add watchdog to detect hangs

7. **Add Docker Health Check Configuration**
   ```yaml
   healthcheck:
     test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
     interval: 10s
     timeout: 5s
     retries: 3
     start_period: 30s
   ```

## Detection Strategies

1. **Check Container Exit Codes**
   - Exit code 137 = OOM kill
   - Exit code 143 = SIGTERM
   - Exit code 1 = General error

2. **Check System Logs**
   ```bash
   dmesg | grep -i "killed process"
   journalctl -u docker
   ```

3. **Add Exit Code Logging**
   ```python
   import atexit
   def log_exit():
       logger.critical(f"Process exiting with code: {sys.exitcode}")
   atexit.register(log_exit)
   ```

4. **Monitor Health Check Response Times**
   - Log health check duration
   - Alert on slow health checks
   - Track health check failures

5. **Add Heartbeat Logging**
   ```python
   async def heartbeat():
       while True:
           logger.info("Heartbeat: service alive")
           await asyncio.sleep(60)
   ```

## Priority Order

1. **High Priority**: Health check timeout wrapper, signal handlers, fallback logging
2. **Medium Priority**: Memory monitoring, lifespan startup wrapping, process monitoring
3. **Low Priority**: Background task monitoring, connection pool limits
