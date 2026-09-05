"""In-memory task storage for 3D reconstruction tasks."""

from datetime import datetime, timezone
from typing import Any

# task_id -> task data dict
_tasks: dict[str, dict[str, Any]] = {}


def create_task(
    task_id: str,
    estimated_time_cost: int,
    video_start_timestamp: int,
) -> dict[str, Any]:
    """Register a new reconstruction task with PENDING status."""
    now = datetime.now(timezone.utc).isoformat()
    task = {
        "task_id": task_id,
        "status": "PENDING",
        "stage": "",
        "result": None,
        "error_message": "",
        "estimated_time_cost": estimated_time_cost,
        "created_at": now,
        "video_start_timestamp": video_start_timestamp,
    }
    _tasks[task_id] = task
    return task


def get_task(task_id: str) -> dict[str, Any] | None:
    """Return task data by ID, or None if not found."""
    return _tasks.get(task_id)


def update_status(
    task_id: str,
    status: str,
    stage: str = "",
    result: dict | None = None,
    error_message: str = "",
) -> dict[str, Any] | None:
    """Update task status fields. Returns updated task or None."""
    task = _tasks.get(task_id)
    if task is None:
        return None
    task["status"] = status
    if stage:
        task["stage"] = stage
    if result is not None:
        task["result"] = result
    task["error_message"] = error_message
    return task
