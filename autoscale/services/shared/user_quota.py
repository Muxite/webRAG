import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Optional

from supabase import Client
from shared.supabase_client import create_user_client, create_service_client


@dataclass
class UserQuotaResult:
    """
    Result of a per-user quota check and consume call.
    """

    allowed: bool
    remaining: Optional[int]


class QuotaManager(ABC):
    """
    Abstract base class for quota management.
    Allows dependency injection for testing.
    """

    @abstractmethod
    def check_and_consume(self, access_token: str, user_id: str, email: str, units: int) -> UserQuotaResult:
        """
        Check and consume quota units.
        :param access_token: User's access token
        :param user_id: User ID
        :param email: User email
        :param units: Number of units to consume
        :returns UserQuotaResult: Result of quota check
        """
        pass


class NoOpQuotaManager(QuotaManager):
    """
    No-op quota manager that always allows all requests.
    Used for testing when quota checks should be disabled.
    """

    def check_and_consume(self, access_token: str, user_id: str, email: str, units: int) -> UserQuotaResult:
        """
        Always allow quota consumption.
        :param access_token: User's access token (unused)
        :param user_id: User ID (unused)
        :param email: User email (unused)
        :param units: Number of units to consume (unused)
        :returns UserQuotaResult: Always returns allowed=True
        """
        return UserQuotaResult(allowed=True, remaining=None)


class SupabaseUserTickManager(QuotaManager):
    """
    Manages per-user daily tick quotas using Supabase with RLS.
    
    Uses user's JWT token to create authenticated clients that respect RLS policies.
    Users can only access their own profiles and usage data.
    """

    def __init__(
        self,
        default_daily_limit: int = 32,
        profile_table: str = "profiles",
        usage_table: str = "user_daily_usage",
    ) -> None:
        """Initialize with default limit and table names."""
        self.default_daily_limit = default_daily_limit
        self.profile_table = profile_table
        self.usage_table = usage_table

    def _get_user_client(self, access_token: str) -> Client:
        """
        Create Supabase client with user's token for RLS.
        :param access_token: User's access token
        :returns Client: Supabase client
        """
        return create_user_client(access_token)

    def _get_service_client(self) -> Optional[Client]:
        """
        Create Supabase client with service role key (bypasses RLS).
        Returns None if service role key is not available.
        :returns Optional[Client]: Supabase client or None
        """
        try:
            return create_service_client()
        except RuntimeError as e:
            if "SUPABASE_SERVICE_ROLE_KEY" in str(e):
                return None
            raise

    def _get_or_create_profile(self, user_id: str, email: str, access_token: str) -> int:
        """
        Get or create user profile.
        Tries service role client first (bypasses RLS), falls back to user client if service role unavailable.
        :param user_id: User ID
        :param email: User email
        :param access_token: User's access token for fallback
        :returns int: Daily tick limit
        """
        service_client = self._get_service_client()
        client = service_client if service_client else self._get_user_client(access_token)
        
        upsert_response = (
            client.table(self.profile_table)
            .upsert(
                {
                    "user_id": user_id,
                    "email": email,
                    "daily_tick_limit": self.default_daily_limit,
                }
            )
            .execute()
        )

        upsert_error = getattr(upsert_response, "error", None)
        if upsert_error:
            raise RuntimeError(f"Failed to upsert profile: {upsert_error}")

        # Fetch the profile to get the actual daily_tick_limit (may differ from default if it existed)
        response = (
            client.table(self.profile_table)
            .select("daily_tick_limit")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )

        data = getattr(response, "data", None)
        error = getattr(response, "error", None)

        if error:
            raise RuntimeError(f"Failed to load profile after upsert: {error}")

        if data:
            limit = data.get("daily_tick_limit")
            if isinstance(limit, int):
                return limit

        return self.default_daily_limit

    def _get_usage_row(self, client: Client, user_id: str, today: date) -> dict | None:
        """Get usage row for user and date. RLS ensures users can only access their own data."""
        response = (
            client.table(self.usage_table)
            .select("id, ticks_used")
            .eq("user_id", user_id)
            .eq("usage_date", today.isoformat())
            .maybe_single()
            .execute()
        )

        data = getattr(response, "data", None)
        error = getattr(response, "error", None)

        if error:
            raise RuntimeError(f"Failed to load usage row: {error}")
        return data

    def check_and_consume(self, access_token: str, user_id: str, email: str, units: int) -> UserQuotaResult:
        """
        Check and consume ticks.
        Uses service role client for profile operations (bypasses RLS).
        Uses user client for usage operations (respects RLS).
        :param access_token: User's access token
        :param user_id: User ID
        :param email: User email
        :param units: Number of ticks to consume
        :returns UserQuotaResult: Result of quota check
        """
        if units <= 0:
            return UserQuotaResult(allowed=True, remaining=None)

        user_client = self._get_user_client(access_token)

        today = date.today()
        limit = self._get_or_create_profile(user_id, email, access_token)
        if limit <= 0:
            return UserQuotaResult(allowed=True, remaining=None)

        current_row = self._get_usage_row(user_client, user_id, today)
        current_used = int(current_row.get("ticks_used", 0)) if current_row else 0
        new_used = current_used + units

        if new_used > limit:
            remaining = max(0, limit - current_used)
            return UserQuotaResult(allowed=False, remaining=remaining)

        if current_row:
            update_response = (
                user_client.table(self.usage_table)
                .update({"ticks_used": new_used})
                .eq("id", current_row["id"])
                .execute()
            )
            update_error = getattr(update_response, "error", None)
            if update_error:
                raise RuntimeError(f"Failed to update usage: {update_error}")
        else:
            insert_response = (
                user_client.table(self.usage_table)
                .insert(
                    {
                        "user_id": user_id,
                        "usage_date": today.isoformat(),
                        "ticks_used": new_used,
                    }
                )
                .execute()
            )
            insert_error = getattr(insert_response, "error", None)
            if insert_error:
                raise RuntimeError(f"Failed to insert usage: {insert_error}")

        remaining = max(0, limit - new_used)
        return UserQuotaResult(allowed=True, remaining=remaining)


