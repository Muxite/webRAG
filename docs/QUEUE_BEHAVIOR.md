# Queue Behavior and Message Processing

## Normal Operation

### Queue Depth Expectations

**Queue depth of 0 is normal and expected** when:
- Agent service is running (1/1 tasks)
- Agent is connected to RabbitMQ
- Messages are being processed successfully

**Important**: RabbitMQ's `message_count` (used by queue depth) only counts **unacknowledged messages in the queue**. Once a message is consumed by the agent, it's removed from the queue count even if it's still being processed. This means:
- Queue depth 0 = No messages waiting in queue
- Messages may still be "pending" in frontend if agent hasn't started processing them yet
- With sequential processing, queue depth will be 0-1 during normal operation

### Why Queue Depth is 0

**RabbitMQ queue depth behavior**:
- `message_count` only counts **unacknowledged messages in the queue**
- Agent uses prefetch=1, so only one message is delivered at a time
- Once agent consumes a message, it's removed from queue count immediately
- Message is acknowledged when processing completes (via `message.process()` context manager)

The agent processes messages **one at a time**:
1. Agent consumes message from queue → **removed from queue count**
2. Agent processes message (takes time, e.g., 10s for skip messages)
3. Message is acknowledged when processing completes
4. Next message is consumed when current one completes

**Result**: With sequential processing:
- Queue depth: **0** (messages consumed immediately)
- Frontend "pending": Tasks waiting to be processed or status not yet updated
- Frontend "in_progress": Task currently being processed

This is **normal behavior** - queue depth 0 doesn't mean no work is happening!

### Skip Message Processing Time

Skip messages (`skipskipskip`) complete quickly:
- **Connectivity test**: ~1-2 seconds
- **Wait delay**: 10 seconds (configurable via `AGENT_SKIP_DELAY_SECONDS`)
- **Total**: ~10-12 seconds per message

**Example**: 30 skip messages
- Processing time: 30 × 10s = 300s (5 minutes)
- Queue depth during processing: 0-1
- Queue depth after completion: 0

## Verifying Message Processing

### Check Agent Logs

Look for "SKIP MODE" entries in agent logs:
```
[INFO] [Agent] SKIP MODE: Mandate contains skip phrase 'skipskipskip'
[INFO] [Agent] SKIP MODE: Waiting 10s to allow queue to fill up for testing...
[INFO] [Agent] SKIP MODE: Connectivity test complete
```

### Check Gateway Logs

Look for task submissions:
```
Task submitted: correlation_id=..., mandate=skipskipskip
```

### Check Queue Depth Over Time

Monitor metrics service logs:
```
[INFO] [MetricsService] Queue depth: agent.mandates=0
```

If queue depth stays at 0 and you see agent activity, messages are being processed successfully.

## Troubleshooting

### Queue Depth 0 with Pending Tasks

**Symptom**: Queue depth is 0, but frontend shows tasks as "pending".

**Explanation**: This is normal! RabbitMQ queue depth only counts unacknowledged messages. When agent consumes a message:
1. Message is removed from queue count immediately
2. Agent processes message (takes time)
3. Message is acknowledged when processing completes

So queue depth 0 means:
- Messages are being consumed by agent
- Agent is processing them sequentially
- Frontend "pending" status updates when agent publishes status to Redis

**Verification**: Check agent logs for "SKIP MODE" or processing activity.

### Queue Depth Always 0, No Agent Logs

**Symptom**: Queue depth is 0, but no agent processing logs appear.

**Possible Causes**:
1. Agent not running or not connected to RabbitMQ
2. Agent cannot resolve service discovery DNS
3. Agent health check failing (check agent service status)

**Fix**: Check agent service status and connectivity:
```bash
python scripts/check-autoscale.py
python scripts/diagnose-agent-connectivity.py
```

### Queue Depth Growing

**Symptom**: Queue depth increases over time (2, 3, 4+ messages).

**Possible Causes**:
1. Agent processing slower than message arrival
2. Agent crashed or unhealthy
3. Messages taking longer than expected

**Fix**: 
- Check agent service: `python scripts/check-autoscale.py`
- Check agent logs for errors
- Consider scaling agent service if load is high

### Messages Not Being Queued

**Symptom**: Submitted messages but queue depth stays 0, no agent logs.

**Possible Causes**:
1. Gateway not publishing to RabbitMQ
2. Wrong queue name
3. RabbitMQ connection issues

**Fix**: Check gateway logs for publishing errors.

## Configuration

### Agent Processing Rate

- **Concurrent messages**: 1 (agent processes sequentially)
- **RabbitMQ prefetch**: 1 (only one unacknowledged message at a time)
- **Message acknowledgment**: Automatic after processing completes
- **Skip message delay**: 10 seconds (configurable via `AGENT_SKIP_DELAY_SECONDS`)
- **Normal message processing**: Varies by mandate complexity

**Important**: The agent uses RabbitMQ QoS prefetch=1, ensuring:
- Only one message is delivered to the agent at a time
- Message is acknowledged only after processing completes
- Next message is delivered only after current message is acknowledged

### Queue Monitoring

- **Metrics interval**: 5 seconds (configurable via `QUEUE_DEPTH_METRICS_INTERVAL`)
- **CloudWatch namespace**: `Euglena/RabbitMQ`
- **Metric name**: `QueueDepth`

## Understanding "Pending" vs Queue Depth

### Scenario: Queue Depth 0, 13 Tasks Pending

This is **normal**! Here's what's happening:

1. **18 tasks completed**: Agent processed and completed these
2. **Task 19 in_progress**: Agent is currently processing this
3. **Tasks 20-32 pending**: These have been consumed from queue but:
   - Agent processes sequentially (one at a time)
   - Frontend shows "pending" until agent publishes status update
   - Queue depth is 0 because messages are consumed immediately

**Timeline**:
- T=0s: 32 messages in queue
- T=1s: Agent consumes all 32 messages → queue depth = 0
- T=1-180s: Agent processes messages sequentially (18 completed, 1 in_progress, 13 waiting)
- T=180-300s: Remaining 13 messages processed

**Key Point**: Queue depth 0 means messages are consumed, not that processing is complete.

## Key Takeaways

1. **Queue depth 0 = Normal**: Messages consumed by agent (may still be processing)
2. **Pending tasks = Normal**: Agent processing sequentially, status updates pending
3. **Queue depth 0 + No logs = Bad**: Agent may not be running
4. **Queue depth growing = Warning**: Processing lag or agent issues
5. **Skip messages process quickly**: ~10 seconds each
6. **Single agent = Sequential processing**: One message at a time
7. **Queue depth ≠ Processing status**: Depth shows queue, not processing state
