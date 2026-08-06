"""Attribution-style JSONL summary row, one per completed mandate.

Deliberately NOT forced into badmodel-lab/results/cells.jsonl's `run_id`/`ids`/`runs`
shape — those encode a *benchmark cell* (a known task id run N times); a friend's
freeform chat mandate isn't that, and fabricating those fields would misrepresent the
data. Field names that DO overlap (`model`, `profile`, `tier`) are deliberate, so a
future ingestion script could normalize both into one analysis frame.

Fields are read from IdeaDagEngine's actual final payload
(agent.app.idea_finalize.build_final_payload / IdeaDagEngine.finalize) — NOT from
Agent.metrics, which stays at its all-zero __init__ default whenever the idea-dag path
runs (the manual tick-loop that updates Agent.metrics is skipped entirely; confirmed by
reading agent.app.agent.Agent.run()).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

LOG_DIR = Path("/app/playground_logs")
SUMMARY_PATH = LOG_DIR / "session_summary.jsonl"


def append_summary_row(
    *,
    session_id: str,
    mandate_id: str,
    tier: str,
    model: Optional[str],
    profile: str,
    mandate_text: str,
    started_at: float,
    result: Dict[str, Any],
    transcript_file: str,
) -> None:
    now = time.time()
    row = {
        "session_id": session_id,
        "mandate_id": mandate_id,
        "tier": tier,
        "model": model,
        "profile": profile,
        "ts_start": started_at,
        "ts_end": now,
        "duration_s": round(now - started_at, 2),
        "mandate_preview": (mandate_text or "")[:200],
        "success": result.get("success"),
        "goal_achieved": result.get("goal_achieved"),
        "has_failures": result.get("has_failures"),
        "grounded": result.get("grounded"),
        "sources_count": len(result.get("sources") or []),
        "pending_nodes_count": result.get("pending_nodes_count"),
        "deliverable_chars": len(result.get("final_deliverable") or ""),
        "transcript_file": transcript_file,
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
