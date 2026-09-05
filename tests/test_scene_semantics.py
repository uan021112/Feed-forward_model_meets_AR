from pathlib import Path

import numpy as np

from app.services.scene_semantics import build_scene_objects
from app.services.storage import get_objects_path


def test_build_scene_objects_persists_stable_ids_and_language_features(tmp_path: Path):
    task_root = tmp_path / "task"
    # Use the explicit temporary task root expected by the service paths.
    (task_root / "workspace" / "semantic").mkdir(parents=True)
    rng = np.random.default_rng(7)
    first = rng.normal([0.0, 0.0, 1.0], 0.005, size=(64, 3))
    second = rng.normal([0.5, 0.0, 1.0], 0.005, size=(64, 3))
    points = np.concatenate([first, second]).astype(np.float32)
    part_features = np.concatenate([np.tile([1.0, 0.0], (64, 1)), np.tile([0.0, 1.0], (64, 1))]).astype(np.float32)
    confidence = np.ones(128, dtype=np.float32)
    lseg_features = np.ones((128, 512), dtype=np.float32)
    lseg_features /= np.linalg.norm(lseg_features, axis=1, keepdims=True)
    payload = build_scene_objects(task_root, points, part_features, confidence, points, lseg_features)
    assert [obj["object_id"] for obj in payload["objects"]] == ["object_0001", "object_0002"]
    assert all(len(obj["language_feature"]) == 512 for obj in payload["objects"])
    assert get_objects_path(task_root).exists()
