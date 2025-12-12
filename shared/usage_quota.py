import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from shared.connector_config import ConnectorConfig
from shared.connector_redis import ConnectorRedis


@dataclass
class QuotaResult:
    allowed: bool
    remaining: Optional[int]


class UsageQuotaBackend(ABC):
    """Abstract daily usage quota backend.

    Implementations should track and enforce a global daily usage counter.
    The unit is an arbitrary integer (we use agent ticks). The interface is
    intentionally small to keep it easily replaceable.
    """

    @abstractmethod
    async def check_and_consume(self, units: int) -> QuotaResult:
        """Atomically check remaining quota for today and consume `units` if possible."""
        raise NotImplementedError

    @abstractmethod
    async def get_usage(self) -> int:
        """Return current accumulated usage for today (best-effort)."""
        raise NotImplementedError


def _seconds_until_end_of_day_utc(now: Optional[datetime] = None) -> int:
    now = now or datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).date()
    end = datetime.combine(tomorrow, datetime.min.time(), tzinfo=timezone.utc)
    return max(1, int((end - now).total_seconds()))


class InMemoryDailyQuota(UsageQuotaBackend):
    """Process-local in-memory quota for development/testing.

    Not suitable for multi-process or multi-instance deployments.
    """

    def __init__(self, limit_per_day: int):
        self.limit = limit_per_day
        self._usage = 0
        self._day = datetime.utcnow().date()
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger(self.__class__.__name__)

    async def _rollover_if_needed(self):
        today = datetime.utcnow().date()
        if today != self._day:
            self._day = today
            self._usage = 0

    async def check_and_consume(self, units: int) -> QuotaResult:
        if self.limit <= 0:
            return QuotaResult(True, None)
        async with self._lock:
            await self._rollover_if_needed()
            if self._usage + units > self.limit:
                remaining = max(0, self.limit - self._usage)
                return QuotaResult(False, remaining)
            self._usage += units
            remaining = self.limit - self._usage
            return QuotaResult(True, remaining)

    async def get_usage(self) -> int:
        async with self._lock:
            await self._rollover_if_needed()
            return self._usage


class RedisDailyQuota(UsageQuotaBackend):
    """Redis-based global daily quota using a date key and INCRBY.

    Key pattern: quota:daily:YYYYMMDD
    We set an expire to end-of-day to auto-reset.
    """

    def __init__(self, config: Optional[ConnectorConfig] = None):
        self.config = config or ConnectorConfig()
        self.limit = self.config.daily_tick_limit
        self.connector = ConnectorRedis(self.config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._prefix = "quota:daily:"

    def _key(self) -> str:
        today = datetime.utcnow().strftime("%Y%m%d")
        return f"{self._prefix}{today}"

    async def check_and_consume(self, units: int) -> QuotaResult:
        if self.limit <= 0:
            return QuotaResult(True, None)

        async with self.connector as conn:
            client = await conn.get_client()
            if client is None:
                # fail-open to avoid blocking service if Redis is down
                self.logger.warning("Redis unavailable, quota check bypassed (fail-open)")
                return QuotaResult(True, None)

            key = self._key()
            # Ensure key exists and has TTL to end of day
            ttl = await client.ttl(key)
            if ttl is None or ttl < 0:
                # Initialize to 0 with expire
                await client.set(key, 0, ex=_seconds_until_end_of_day_utc())

            # Optimistic check then incr in a Lua script for atomicity
            script = """
            local key = KEYS[1]
            local limit = tonumber(ARGV[1])
            local units = tonumber(ARGV[2])
            local current = tonumber(redis.call('GET', key) or '0')
            if (current + units) > limit then
              return {0, limit - current}
            end
            local newv = redis.call('INCRBY', key, units)
            return {1, limit - newv}
            """
            try:
                allowed, remaining = await client.eval(script, numkeys=1, keys=[key], args=[self.limit, units])
                # remaining may be negative briefly; clamp at >=0
                remaining = max(0, int(remaining)) if remaining is not None else 0
                return QuotaResult(bool(allowed), remaining)
            except Exception as e:
                self.logger.error(f"Redis quota script failed: {e}")
                # fail-open
                return QuotaResult(True, None)

    async def get_usage(self) -> int:
        async with self.connector as conn:
            client = await conn.get_client()
            if client is None:
                return 0
            val = await client.get(self._key())
            try:
                return int(val) if val is not None else 0
            except Exception:
                return 0
