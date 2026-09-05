import json

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.services.storage import create_task_dirs, get_object_content_path, get_objects_path
from app.services.task_store import _tasks, create_task, update_status


def _client(monkeypatch, tmp_path):
    async def no_op_background(*_args, **_kwargs):
        return None
    _tasks.clear()
    monkeypatch.setenv("RECONSTRUCT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("IGGT_PRELOAD_ON_STARTUP", "0")
    monkeypatch.setenv("LSEG_PRELOAD_ON_STARTUP", "0")
    monkeypatch.setattr("app.routers.reconstruct._process_task_background", no_op_background)
    return TestClient(app)


def _jpg_bytes():
    ok, encoded = cv2.imencode(".jpg", np.full((12, 16, 3), 180, dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def test_submit_reconstruction_accepts_optional_ar_pose(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    response = client.post("/api/v1/reconstruct", data={"video_start_timestamp": "0"}, files={"video_file": ("capture.mp4", b"fake", "video/mp4")})
    assert response.status_code == 202
    assert response.json()["data"]["has_ar_pose"] is False


def test_submit_reconstruction_rejects_invalid_ar_pose(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    response = client.post("/api/v1/reconstruct", data={"video_start_timestamp": "0"}, files={"video_file": ("capture.mp4", b"fake", "video/mp4"), "ar_pose_file": ("pose.csv", b"bad", "text/csv")})
    assert response.status_code == 400


def test_object_content_is_persisted_and_returned(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    task_id = "rec_objects"
    create_task(task_id=task_id, estimated_time_cost=1, video_start_timestamp=0)
    update_status(task_id, status="SUCCESS", stage="Completed", result={"scene_artifacts_ready": True})
    root = create_task_dirs(task_id)
    get_objects_path(root).write_text(json.dumps({"format_version": 1, "objects": [{"object_id": "object_0001", "centroid": {"x": 0, "y": 0, "z": 0}}]}))
    response = client.put(f"/api/v1/task/{task_id}/objects/object_0001/content", json={"annotation": {"text": "engine"}})
    assert response.status_code == 200
    assert response.json()["content"]["annotation"]["text"] == "engine"
    assert json.loads(get_object_content_path(root).read_text())["objects"]["object_0001"]["annotation"]["text"] == "engine"
