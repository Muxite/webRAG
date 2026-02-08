# Audit Scripts Consolidation

**Date**: 2026-02-05  
**Action**: Combined 3 audit scripts into one comprehensive tool

## Previous Scripts

1. **`audit-aws-changes.py`** - Basic audit using AWSAuditor
2. **`analyze-audit.py`** - Analyzes JSON output for breaking points
3. **`aws_audit.py`** - Core AWSAuditor class (kept as module)

## New Unified Script

**`scripts/comprehensive-audit.py`** - Combines all functionality plus adds:

### New Capabilities

1. **Task-Level Investigation**
   - Get task details with container exit codes
   - Analyze task definition configurations
   - Check target group health
   - Find recent failed tasks

2. **RabbitMQ-Specific Analysis**
   - Detects OOM kills (exit code 137)
   - Calculates memory per container
   - Identifies RabbitMQ container issues
   - Provides specific recommendations

3. **Comprehensive Analysis**
   - Combines audit data with task investigation
   - Correlates task failures with task definition changes
   - Identifies resource changes that could cause issues

### Usage

```bash
# Investigate specific task
python scripts/comprehensive-audit.py \
  --task-arn "arn:aws:ecs:us-east-2:848960888155:task/euglena-cluster/cbb8903ef5264c9b9621ff182ab9c6a2" \
  --target-group-arn "arn:aws:elasticloadbalancing:us-east-2:848960888155:targetgroup/euglena-tg/ea4bbe2f98578c2a" \
  --days 3 \
  --output rabbitmq-investigation.json

# General audit
python scripts/comprehensive-audit.py --days 3 --output audit.json

# Recent failures only
python scripts/comprehensive-audit.py --days 1
```

### Features

- **Error Handling**: Gracefully handles CloudTrail/auth errors
- **Task Investigation**: Deep dive into specific task failures
- **Resource Analysis**: Calculates memory per container
- **RabbitMQ Detection**: Specifically identifies RabbitMQ issues
- **Recommendations**: Provides actionable fixes

## Migration

The old scripts can be deprecated:
- `audit-aws-changes.py` → Use `comprehensive-audit.py`
- `analyze-audit.py` → Functionality integrated into `comprehensive-audit.py`
- `aws_audit.py` → Kept as reusable module

## Benefits

1. **Single tool** for all audit needs
2. **No repetition** - code reused from AWSAuditor
3. **Task-level insights** - can investigate specific failures
4. **Better error handling** - continues even if some APIs fail
5. **RabbitMQ-specific** - tailored analysis for RabbitMQ issues
