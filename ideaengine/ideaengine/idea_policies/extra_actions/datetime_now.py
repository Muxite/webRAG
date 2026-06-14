"""datetime_now action — anchor the current moment for time-sensitive prompts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from ideaengine.idea_policies.actions import LeafAction
from ideaengine.idea_policies.extra_actions.base import fail, ok


class DatetimeNowAction(LeafAction):
    """Return the current date / time, with optional offset.

    Reads from node details:
      - `tz_offset_hours` (number): optional UTC offset (e.g. -5 for EST).
      - `add_days` / `add_hours` / `add_minutes`: optional time-shift.

    Returns `{iso, date, time, weekday, unix_seconds, tz_offset_hours}`.
    """

    name = "datetime_now"

    async def execute(self, graph, node_id: str, io: Any) -> Dict[str, Any]:
        node = graph.get_node(node_id)
        if not node:
            return fail(self.name, f"node {node_id} not found")
        details = node.details or {}
        try:
            offset_hours = float(details.get("tz_offset_hours") or 0.0)
            add_days = float(details.get("add_days") or 0.0)
            add_hours = float(details.get("add_hours") or 0.0)
            add_minutes = float(details.get("add_minutes") or 0.0)
        except (TypeError, ValueError) as exc:
            return fail(self.name, f"invalid numeric detail: {exc}")

        tz = timezone(timedelta(hours=offset_hours))
        now = datetime.now(tz)
        shifted = now + timedelta(days=add_days, hours=add_hours, minutes=add_minutes)
        return ok(
            self.name,
            iso=shifted.isoformat(),
            date=shifted.date().isoformat(),
            time=shifted.time().isoformat(timespec="seconds"),
            weekday=shifted.strftime("%A"),
            unix_seconds=shifted.timestamp(),
            tz_offset_hours=offset_hours,
        )
