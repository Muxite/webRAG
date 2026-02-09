import logging
from datetime import datetime
from typing import Optional

from shared.storage import RedisTaskStorage, SupabaseTaskStorage
from shared.models import TaskRequest
from shared.message_contract import TaskState, StatusType
from shared.supabase_client import create_user_client


class GatewayTaskRegistrar:
    """
    Coordinate task registration between Redis status updates and Supabase records.
    """

    def __init__(self, redis_storage: RedisTaskStorage, supabase_storage: SupabaseTaskStorage) -> None:
        """
        Initialize the registrar with Redis and Supabase storage backends.
        :param redis_storage: Redis task storage used by workers
        :param supabase_storage: Supabase task storage for persisted history
        :return: None
        """
        self.redis_storage = redis_storage
        self.supabase_storage = supabase_storage
        self.logger = logging.getLogger(self.__class__.__name__)

    def _normalize_status(self, value: Optional[str]) -> str:
        """
        Normalize worker status values into Supabase task status values.
        :param value: Raw status string
        :return: Normalized status string
        """
        if not value:
            return TaskState.IN_PROGRESS.value
        if value in (StatusType.ACCEPTED.value, StatusType.STARTED.value, StatusType.IN_PROGRESS.value):
            return TaskState.IN_PROGRESS.value
        if value == StatusType.COMPLETED.value:
            return TaskState.COMPLETED.value
        if value == StatusType.ERROR.value:
            return TaskState.FAILED.value
        if value == TaskState.PENDING.value:
            return "in_queue"
        return value

    async def register_new_task(self, user_id: str, access_token: str, req: TaskRequest, correlation_id: str) -> dict:
        """
        Insert an initial Supabase task record for a new submission.
        :param user_id: Supabase auth user id
        :param access_token: Supabase JWT token for RLS
        :param req: Task submission request
        :param correlation_id: Task correlation id
        :return: Insert payload
        """
        now = datetime.utcnow().isoformat()
        payload = {
            "correlation_id": correlation_id,
            "user_id": user_id,
            "mandate": req.mandate,
            "status": "in_queue",
            "max_ticks": int(req.max_ticks or 50),
            "tick": None,
            "result": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }
        user_client = create_user_client(access_token)

        def _do():
            return user_client.table(self.supabase_storage.table).insert(payload).execute()

        response = await self.supabase_storage._execute(_do)
        error = getattr(response, "error", None)
        if error:
            raise RuntimeError(f"Failed to insert task: {error}")
        self.logger.info(
            "Supabase task registered",
            extra={"correlation_id": correlation_id, "user_id": user_id, "status": payload["status"]},
        )
        return payload

    async def _update_supabase(self, correlation_id: str, updates: dict) -> bool:
        """
        Apply updates in Supabase and return whether a row was updated.
        :param correlation_id: Task correlation id
        :param updates: Update payload
        :return: True if a row was updated
        """
        def _do():
            return (
                self.supabase_storage.client
                .table(self.supabase_storage.table)
                .update(updates)
                .eq("correlation_id", correlation_id)
                .execute()
            )

        response = await self.supabase_storage._execute(_do)
        error = getattr(response, "error", None)
        if error:
            raise RuntimeError(f"Failed to update task {correlation_id}: {error}")
        data = getattr(response, "data", None)
        return bool(data)

    async def sync_from_redis_once(self) -> int:
        """
        Sync Redis task statuses into Supabase and prune terminal tasks from Redis.
        :return: Count of synced tasks
        """
        tasks = await self.redis_storage.list_tasks()
        synced = 0
        deleted = 0
        missing = 0
        for data in tasks:
            if not isinstance(data, dict):
                continue
            correlation_id = data.get("correlation_id")
            if not correlation_id:
                continue
            status = self._normalize_status(data.get("status"))
            updates = {
                "status": status,
                "updated_at": data.get("updated_at") or datetime.utcnow().isoformat(),
            }
            if "mandate" in data:
                updates["mandate"] = data.get("mandate")
            if "tick" in data:
                updates["tick"] = data.get("tick")
            if "max_ticks" in data:
                updates["max_ticks"] = data.get("max_ticks")
            if "result" in data:
                updates["result"] = data.get("result")
            if "error" in data:
                updates["error"] = data.get("error")

            try:
                updated = await self._update_supabase(correlation_id, updates)
                if not updated:
                    self.logger.warning(
                        "Supabase task row not found for Redis update",
                        extra={"correlation_id": correlation_id},
                    )
                    missing += 1
                    continue
                synced += 1
                self.logger.info(
                    "Supabase task updated",
                    extra={"correlation_id": correlation_id, "status": updates.get("status")},
                )
            except Exception as e:
                self.logger.warning(
                    "Supabase task update failed",
                    extra={"correlation_id": correlation_id, "error": str(e)},
                )
                continue

            if status in (TaskState.COMPLETED.value, TaskState.FAILED.value):
                try:
                    await self.redis_storage.delete_task(correlation_id)
                    deleted += 1
                    self.logger.info(
                        "Redis task removed after terminal status",
                        extra={"correlation_id": correlation_id, "status": status},
                    )
                except Exception as e:
                    self.logger.debug(
                        "Failed to delete Redis task after completion",
                        extra={"correlation_id": correlation_id, "error": str(e)},
                    )
        if tasks:
            self.logger.info(
                "Redis status sync complete",
                extra={"seen": len(tasks), "synced": synced, "deleted": deleted, "missing": missing},
            )
        return synced

