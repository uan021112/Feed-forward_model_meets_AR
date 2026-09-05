"""IGGT runtime pose refinement and confidence-aware reference maintenance."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from app.config import (
    get_reference_connection_threshold,
    get_reference_max_views,
    get_reference_min_confidence,
    get_reference_min_connections,
    get_reference_novelty_weight,
    get_reference_reliability_weight,
    get_reference_retrieval_count,
    get_reference_reward_margin,
    get_runtime_alignment_max_rms,
)
from app.services.geometry import atomic_json_write, weighted_sim3_alignment
from app.services.iggt import run_iggt
from app.services.storage import get_frame_images_dir, get_mesh_dir, get_reconstructed_pose_dir, get_relocalization_dir, get_relocalization_intrinsics_path, get_transforms_path
from app.utils.image import decode_image_bytes
from app.utils.pose import matrix_to_quaternion

_MIN_REFERENCE_VIEWS = 3
_POOL_LOCKS: dict[str, threading.Lock] = {}
_POOL_LOCKS_LOCK = threading.Lock()


class RelocalizationUnavailableError(RuntimeError):
    pass


class RelocalizationInputError(ValueError):
    pass


decode_query_image = decode_image_bytes


@dataclass(frozen=True)
class ReferenceView:
    view_id: str
    image_path: Path
    c2w: np.ndarray
    intrinsics: np.ndarray
    image_size: tuple[int, int]
    timestamp: int | None
    reliability: float
    descriptor: np.ndarray
    visible_faces: frozenset[int]


def _pool_lock(task_root: Path) -> threading.Lock:
    with _POOL_LOCKS_LOCK:
        return _POOL_LOCKS.setdefault(str(task_root.resolve()), threading.Lock())


def _validate_pose(pose: np.ndarray) -> None:
    if np.asarray(pose).shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-4) or not np.isclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-2):
        raise ValueError("Invalid camera pose")


def _validate_intrinsics(K: np.ndarray, size: tuple[int, int]) -> None:
    if np.asarray(K).shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0 or size[0] <= 0 or size[1] <= 0:
        raise ValueError("Invalid camera intrinsics")


def _descriptor(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).reshape(-1).astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_hist = cv2.resize(edges, (16, 16), interpolation=cv2.INTER_AREA).reshape(-1).astype(np.float32) / 255
    value = np.concatenate([hist / max(float(np.linalg.norm(hist)), 1e-12), edge_hist])
    return value / max(float(np.linalg.norm(value)), 1e-12)


def _visible_faces(task_root: Path, c2w: np.ndarray, K: np.ndarray, size: tuple[int, int]) -> frozenset[int]:
    mesh_path = get_mesh_dir(task_root) / "mesh.glb"
    if not mesh_path.exists():
        return frozenset()
    mesh = trimesh.load(mesh_path, force="mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        return frozenset()
    w2c = np.linalg.inv(c2w)
    centers = vertices[faces].mean(1)
    camera = centers @ w2c[:3, :3].T + w2c[:3, 3]
    width, height = size
    valid = camera[:, 2] > 0
    pixels = (camera @ K.T)
    pixels[:, 0] /= np.maximum(pixels[:, 2], 1e-12)
    pixels[:, 1] /= np.maximum(pixels[:, 2], 1e-12)
    valid &= (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    return frozenset(np.flatnonzero(valid).astype(int).tolist())


def _read_views(task_root: Path) -> list[ReferenceView]:
    images_dir, pose_dir = get_frame_images_dir(task_root), get_reconstructed_pose_dir(task_root)
    intrinsics_path = get_relocalization_intrinsics_path(task_root)
    if not intrinsics_path.exists():
        raise RelocalizationUnavailableError("Scene intrinsics are missing")
    with np.load(intrinsics_path, allow_pickle=False) as payload:
        frame_ids = [str(v) for v in payload["frame_ids"].tolist()]
        Ks = np.asarray(payload["intrinsics"], dtype=np.float64)
        sizes = np.asarray(payload["image_sizes"], dtype=np.int64)
    records = {frame_id: (Ks[i], (int(sizes[i, 0]), int(sizes[i, 1]))) for i, frame_id in enumerate(frame_ids)}
    confidence_path = get_relocalization_dir(task_root) / "view_confidence.npz"
    reliability_by_id: dict[str, float] = {}
    if confidence_path.exists():
        with np.load(confidence_path, allow_pickle=False) as confidence_payload:
            reliability_by_id = {str(view_id): float(value) for view_id, value in zip(confidence_payload["frame_ids"].tolist(), confidence_payload["reliability"].tolist())}
    views = []
    for image_path in sorted(images_dir.glob("*.jpg")):
        record = records.get(image_path.stem)
        pose_path = pose_dir / f"{image_path.stem}.txt"
        if record is None or not pose_path.exists():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        K, size = record
        try:
            w2c = np.loadtxt(pose_path).reshape(4, 4)
            _validate_pose(w2c); _validate_intrinsics(K, size)
        except (OSError, ValueError, np.linalg.LinAlgError):
            continue
        views.append(ReferenceView(image_path.stem, image_path, np.linalg.inv(w2c), K, size, int(image_path.stem) if image_path.stem.isdigit() else None, reliability_by_id.get(image_path.stem, 1.0), _descriptor(image), _visible_faces(task_root, np.linalg.inv(w2c), K, size)))
    if len(views) < _MIN_REFERENCE_VIEWS:
        raise RelocalizationUnavailableError("Fewer than three valid reference views")
    return views


def _view_json(view: ReferenceView) -> dict[str, Any]:
    return {"view_id": view.view_id, "image_path": str(view.image_path), "c2w": view.c2w.tolist(), "intrinsics": view.intrinsics.tolist(), "image_size": list(view.image_size), "timestamp": view.timestamp, "reliability": view.reliability, "descriptor": view.descriptor.tolist(), "visible_faces": sorted(view.visible_faces)}


def _json_view(data: dict[str, Any]) -> ReferenceView:
    return ReferenceView(str(data["view_id"]), Path(data["image_path"]), np.asarray(data["c2w"], dtype=np.float64), np.asarray(data["intrinsics"], dtype=np.float64), tuple(data["image_size"]), data.get("timestamp"), float(data.get("reliability", 1.0)), np.asarray(data.get("descriptor", []), dtype=np.float32), frozenset(int(v) for v in data.get("visible_faces", [])))


def _pool_path(task_root: Path) -> Path:
    return get_relocalization_dir(task_root) / "reference_views.json"


def _load_pool(task_root: Path) -> list[ReferenceView] | None:
    path = _pool_path(task_root)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return [_json_view(item) for item in payload.get("views", [])]


def _save_pool(task_root: Path, views: list[ReferenceView]) -> None:
    atomic_json_write(_pool_path(task_root), {"format_version": 2, "max_views": get_reference_max_views(), "retrieval_count": get_reference_retrieval_count(), "views": [_view_json(view) for view in views]})


def _coverage_novelty(candidate: frozenset[int], existing: list[ReferenceView]) -> float:
    if not existing:
        return 1.0
    return 1.0 - max((_iou(candidate, view.visible_faces) for view in existing), default=0.0)


def _iou(a: frozenset[int], b: frozenset[int]) -> float:
    union = len(a | b)
    return float(len(a & b) / union) if union else 0.0


def initialize_reference_view_pool(task_root: Path) -> int:
    with _pool_lock(task_root):
        existing = _load_pool(task_root)
        if existing is not None:
            return len(existing)
        candidates = _read_views(task_root)
        selected: list[ReferenceView] = []
        remaining = candidates[:]
        capacity = get_reference_max_views()
        while remaining and len(selected) < capacity:
            remaining.sort(key=lambda view: (-get_reference_reliability_weight() * view.reliability - get_reference_novelty_weight() * _coverage_novelty(view.visible_faces, selected), view.view_id))
            selected.append(remaining.pop(0))
        _save_pool(task_root, selected)
        return len(selected)


def _retrieval(query_descriptor: np.ndarray, pool: list[ReferenceView]) -> list[ReferenceView]:
    ranked = sorted(pool, key=lambda view: (-float(view.descriptor @ query_descriptor) if len(view.descriptor) == len(query_descriptor) else 0.0, -view.reliability, view.view_id))
    return ranked[:get_reference_retrieval_count()]


def _set_reward(views: list[ReferenceView]) -> float:
    if not views:
        return 0.0
    return float(sum(get_reference_reliability_weight() * view.reliability + get_reference_novelty_weight() * _coverage_novelty(view.visible_faces, [other for other in views if other is not view]) for view in views))


def _connected(candidate: ReferenceView, pool: list[ReferenceView]) -> bool:
    return sum(_iou(candidate.visible_faces, view.visible_faces) >= get_reference_connection_threshold() for view in pool) >= get_reference_min_connections()


def _update_pool(task_root: Path, candidate: ReferenceView, pool: list[ReferenceView]) -> tuple[list[ReferenceView], bool]:
    if not _connected(candidate, pool):
        return pool, False
    if len(pool) < get_reference_max_views():
        return pool + [candidate], True
    best_reward, remove_index = -float("inf"), None
    baseline = _set_reward(pool)
    for index in range(len(pool)):
        replacement = pool[:index] + pool[index + 1:] + [candidate]
        reward = _set_reward(replacement)
        if reward > best_reward:
            best_reward, remove_index = reward, index
    if remove_index is None or best_reward - baseline <= get_reference_reward_margin():
        return pool, False
    removed = pool[remove_index]
    if removed.image_path.parent == get_relocalization_dir(task_root) / "reference_views":
        removed.image_path.unlink(missing_ok=True)
    return pool[:remove_index] + pool[remove_index + 1:] + [candidate], True


def _pose_result(c2w: np.ndarray) -> dict[str, dict[str, float]]:
    position = c2w[:3, 3]
    quat = matrix_to_quaternion(c2w[:3, :3])
    return {"position": {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])}, "rotation": {"x": float(quat[1]), "y": float(quat[2]), "z": float(quat[3]), "w": float(quat[0])}}


def _root_payload(matrix: np.ndarray) -> dict[str, Any]:
    return {"matrix": matrix.tolist(), "position": {"x": float(matrix[0, 3]), "y": float(matrix[1, 3]), "z": float(matrix[2, 3])}, "rotation": _pose_result(matrix)["rotation"]}


def _commit_runtime_state(task_root: Path, pool: list[ReferenceView], root: np.ndarray | None) -> None:
    _save_pool(task_root, pool)
    if root is not None:
        payload = json.loads(get_transforms_path(task_root).read_text()) if get_transforms_path(task_root).exists() else {"format_version": 1, "coordinate_frame": "G", "metric_scale": True}
        payload["map_to_ar"] = _root_payload(root)
        atomic_json_write(get_transforms_path(task_root), payload)


def relocalize_query_image(task_root: Path, query_image: np.ndarray, *, runtime_c2w: np.ndarray | None = None, query_intrinsics: np.ndarray | None = None, timestamp: int | None = None) -> dict[str, Any]:
    if query_image is None or query_image.size == 0:
        raise RelocalizationInputError("Query image is empty")
    with _pool_lock(task_root):
        pool = _load_pool(task_root) or _read_views(task_root)
        if len(pool) < _MIN_REFERENCE_VIEWS:
            raise RelocalizationUnavailableError("Not enough reference views")
        query_path = get_relocalization_dir(task_root) / f".query_{uuid.uuid4().hex}.jpg"
        if not cv2.imwrite(str(query_path), query_image):
            raise RelocalizationInputError("Failed to stage query image")
        try:
            references = _retrieval(_descriptor(query_image), pool)
            result = run_iggt(get_frame_images_dir(task_root), image_paths=[query_path, *(view.image_path for view in references)])
        finally:
            query_path.unlink(missing_ok=True)
        predicted = np.linalg.inv(result.extrinsics)
        if len(predicted) != len(references) + 1:
            raise RelocalizationUnavailableError("IGGT returned an invalid pose count")
        weights = np.asarray([float(np.nanmean(result.confidence[i])) for i in range(1, len(predicted))], dtype=np.float64)
        rotation, translation, scale, rms, residuals = weighted_sim3_alignment(predicted[1:, :3, 3], np.stack([v.c2w[:3, 3] for v in references]), weights)
        if rms > get_runtime_alignment_max_rms():
            return {"success": False, "reason": "alignment_residual_exceeded", "confidence": 0.0, "alignment_rms_m": rms, "pose": None, "map_to_ar": None, "reference_view_count": len(pool), "reference_view_updated": False}
        refined = predicted[0].copy()
        refined[:3, :3] = rotation @ refined[:3, :3]
        refined[:3, 3] = scale * rotation @ refined[:3, 3] + translation
        confidence = float(np.clip(np.exp(-rms / max(get_runtime_alignment_max_rms(), 1e-6)), 0.0, 1.0) * min(1.0, float(np.mean(weights))))
        if confidence < get_reference_min_confidence():
            return {"success": False, "reason": "confidence_below_threshold", "confidence": confidence, "alignment_rms_m": rms, "pose": None, "map_to_ar": None, "reference_view_count": len(pool), "reference_view_updated": False}
        if runtime_c2w is None:
            return {"success": True, "confidence": confidence, "alignment_rms_m": rms, "pose": _pose_result(refined), "map_to_ar": None, "reference_view_count": len(pool), "reference_view_updated": False, "retrieved_reference_view_count": len(references), "timestamp": timestamp}
        _validate_pose(runtime_c2w)
        root = runtime_c2w @ np.linalg.inv(refined)
        root_payload = _root_payload(root)
        candidate_id = f"query_{uuid.uuid4().hex[:12]}"
        candidate_intrinsics = query_intrinsics if query_intrinsics is not None else result.intrinsics[0]
        if query_intrinsics is None:
            model_height, model_width = result.depths[0].shape[:2]
            sx, sy = query_image.shape[1] / float(model_width), query_image.shape[0] / float(model_height)
            candidate_intrinsics = candidate_intrinsics.copy()
            candidate_intrinsics[0, 0] *= sx
            candidate_intrinsics[0, 2] *= sx
            candidate_intrinsics[1, 1] *= sy
            candidate_intrinsics[1, 2] *= sy
        candidate = ReferenceView(candidate_id, get_relocalization_dir(task_root) / "reference_views" / f"{candidate_id}.jpg", refined, candidate_intrinsics, (query_image.shape[1], query_image.shape[0]), timestamp, float(np.mean(weights)), _descriptor(query_image), _visible_faces(task_root, refined, candidate_intrinsics, (query_image.shape[1], query_image.shape[0])))
        candidate.image_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(candidate.image_path), query_image):
            raise RelocalizationUnavailableError("Failed to persist query reference image")
        updated_pool, changed = _update_pool(task_root, candidate, pool)
        if not changed:
            candidate.image_path.unlink(missing_ok=True)
        _commit_runtime_state(task_root, updated_pool, root)
        return {"success": True, "confidence": confidence, "alignment_rms_m": rms, "pose": _pose_result(refined), "map_to_ar": root_payload, "reference_view_count": len(updated_pool), "reference_view_updated": changed, "retrieved_reference_view_count": len(references), "timestamp": timestamp}
