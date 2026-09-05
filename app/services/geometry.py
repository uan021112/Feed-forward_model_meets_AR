"""Geometry, pose, and reconstruction artifact helpers."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from app.utils.pose import (
    normalize_extrinsics_to_4x4,
    quaternion_to_matrix,
)

CONFIDENCE_PERCENTILE = 10.0


@dataclass(frozen=True)
class ReconstructionFrameSample:
    image_path: Path
    colour_rgb: np.ndarray
    depth: np.ndarray
    depth_intrinsic: np.ndarray
    extrinsic: np.ndarray
    valid_mask: np.ndarray
    v_idx: np.ndarray
    u_idx: np.ndarray
    world_points: np.ndarray


def load_ar_poses(poses_dir: Path, frame_timestamps: list[int]) -> np.ndarray | None:
    """Load interpolated AR world-to-camera matrices in OpenCV RDF coordinates."""
    csv_path = poses_dir / "ar_pose.csv"
    if not csv_path.exists():
        return None
    from app.services.imu_pose import _interpolate_quaternions, _interpolate_vectors

    timestamps: list[int] = []
    positions: list[np.ndarray] = []
    quaternions: list[np.ndarray] = []
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("AR pose CSV is empty")
        for row in reader:
            timestamps.append(int(row["timestamp"]))
            positions.append(np.array([float(row["px"]), float(row["py"]), float(row["pz"])], dtype=np.float32))
            quaternions.append(np.array([float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])], dtype=np.float64))
    if not timestamps:
        raise ValueError("AR pose CSV contains no data rows")
    order = np.argsort(np.asarray(timestamps))
    timestamps = [timestamps[int(i)] for i in order]
    positions = [positions[int(i)] for i in order]
    quaternions = [quaternions[int(i)] for i in order]
    if len(set(timestamps)) != len(timestamps):
        raise ValueError("AR pose CSV contains duplicate timestamps")
    if min(frame_timestamps, default=timestamps[0]) < timestamps[0] or max(frame_timestamps, default=timestamps[-1]) > timestamps[-1]:
        raise ValueError("AR pose timestamps do not cover extracted frames")
    try:
        interp_quats = _interpolate_quaternions(timestamps, quaternions, frame_timestamps)
        interp_positions = _interpolate_vectors(timestamps, positions, frame_timestamps)
    except ValueError as exc:
        raise ValueError(f"AR pose timestamps do not cover extracted frames: {exc}") from exc
    matrices = []
    for q, t in zip(interp_quats, interp_positions):
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = quaternion_to_matrix(np.array([q[3], q[0], q[1], q[2]]))
        c2w[:3, 3] = t[:3]
        matrices.append(np.linalg.inv(c2w))
    return np.stack(matrices).astype(np.float32)


def build_intrinsics(image_paths: list[Path]) -> np.ndarray:
    if not image_paths:
        raise ValueError("Cannot build intrinsics without images")
    values = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        h, w = image.shape[:2]
        values.append(np.array([[w * 0.8, 0, w / 2], [0, h * 0.8, h / 2], [0, 0, 1]], dtype=np.float32))
    return np.stack(values)


def save_reconstruction_intrinsics(image_paths: list[Path], output_path: Path, intrinsics: np.ndarray | None = None, source: str = "iggt_predicted", model_image_size: tuple[int, int] | None = None) -> Path:
    if not image_paths:
        raise ValueError("Cannot save reconstruction intrinsics without images")
    intrinsics = np.asarray(intrinsics if intrinsics is not None else build_intrinsics(image_paths), dtype=np.float32).copy()
    if model_image_size is not None:
        model_width, model_height = model_image_size
        for index, path in enumerate(image_paths):
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot read image: {path}")
            width, height = image.shape[1], image.shape[0]
            sx, sy = width / float(model_width), height / float(model_height)
            intrinsics[index, 0, 0] *= sx
            intrinsics[index, 0, 2] *= sx
            intrinsics[index, 1, 1] *= sy
            intrinsics[index, 1, 2] *= sy
    image_sizes = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        image_sizes.append((image.shape[1], image.shape[0]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp", delete=False) as handle:
        temporary_path = Path(handle.name)
        np.savez_compressed(handle, format_version=np.asarray([2]), frame_ids=np.asarray([p.stem for p in image_paths]), intrinsics=intrinsics, image_sizes=np.asarray(image_sizes, dtype=np.int32), source=np.asarray([source]))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, output_path)
    return output_path


def weighted_sim3_alignment(source_centers: np.ndarray, target_centers: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Solve target ~= scale * R @ source + t with weighted Umeyama SVD."""
    source = np.asarray(source_centers, dtype=np.float64)
    target = np.asarray(target_centers, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) != len(weights):
        raise ValueError("Weighted Sim(3) inputs have incompatible shapes")
    keep = np.isfinite(source).all(1) & np.isfinite(target).all(1) & np.isfinite(weights) & (weights > 0)
    source, target, weights = source[keep], target[keep], weights[keep]
    if len(source) < 3:
        raise ValueError("At least three paired camera centers are required for weighted Sim(3)")
    weight_sum = float(weights.sum())
    source_mean = (source * weights[:, None]).sum(0) / weight_sum
    target_mean = (target * weights[:, None]).sum(0) / weight_sum
    xs = source - source_mean
    yt = target - target_mean
    covariance = (xs * weights[:, None]).T @ yt / weight_sum
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(vt.T @ u.T) < 0:
        correction[-1, -1] = -1
    rotation = vt.T @ correction @ u.T
    variance = float((weights[:, None] * xs * xs).sum() / weight_sum)
    if variance <= 1e-12:
        raise ValueError("Camera centers are degenerate for weighted Sim(3)")
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Weighted Sim(3) produced an invalid scale")
    translation = target_mean - scale * rotation @ source_mean
    residuals = np.linalg.norm(scale * (source @ rotation.T) + translation - target, axis=1)
    rms = float(np.sqrt(np.sum(weights * residuals * residuals) / weight_sum))
    return rotation, translation, scale, rms, residuals


def transform_c2w_by_sim3(c2w: np.ndarray, rotation: np.ndarray, translation: np.ndarray, scale: float) -> np.ndarray:
    pose = np.asarray(c2w, dtype=np.float64)
    transformed = pose.copy()
    transformed[:3, :3] = rotation @ pose[:3, :3]
    transformed[:3, 3] = scale * rotation @ pose[:3, 3] + translation
    return transformed.astype(np.float32)


def transform_w2c_by_sim3(extrinsics: np.ndarray, rotation: np.ndarray, translation: np.ndarray, scale: float) -> np.ndarray:
    normalized = normalize_extrinsics_to_4x4(extrinsics)
    if normalized is None:
        raise ValueError("Cannot transform missing extrinsics")
    return np.stack([np.linalg.inv(transform_c2w_by_sim3(np.linalg.inv(pose), rotation, translation, scale)) for pose in normalized]).astype(np.float32)


def _write_point_cloud_output(point_cloud: o3d.geometry.PointCloud, output_path: Path) -> Path:
    if len(point_cloud.points) == 0:
        raise ValueError("Backprojection produced an empty point cloud")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output_path), point_cloud):
        raise ValueError(f"PLY export failed for {output_path}")
    if len(o3d.io.read_point_cloud(str(output_path)).points) == 0:
        raise ValueError(f"PLY export verification failed for {output_path}")
    return output_path


def _scale_intrinsics_to_depth(intrinsic: np.ndarray, image_width: int, image_height: int, depth_width: int, depth_height: int) -> np.ndarray:
    scaled = np.asarray(intrinsic, dtype=np.float32).copy()
    sx, sy = depth_width / float(image_width), depth_height / float(image_height)
    scaled[0, 0] *= sx; scaled[0, 2] *= sx
    scaled[1, 1] *= sy; scaled[1, 2] *= sy
    return scaled


def iter_reconstruction_frame_samples(images_dir: Path, depths: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray, *, confidence: np.ndarray | None = None, scale: float = 1.0) -> list[ReconstructionFrameSample]:
    normalized = normalize_extrinsics_to_4x4(extrinsics)
    if normalized is None or len(depths) != len(normalized) or len(intrinsics) != len(normalized):
        raise ValueError("Depth, intrinsic, and extrinsic counts must match")
    if confidence is not None and len(confidence) != len(normalized):
        raise ValueError("Confidence, intrinsic, and extrinsic counts must match")
    image_paths = sorted(images_dir.glob("*.jpg"))
    if len(image_paths) < len(depths):
        raise ValueError("Point-cloud export found fewer RGB frames than predictions")
    thresholds = np.percentile(confidence.reshape(len(confidence), -1), CONFIDENCE_PERCENTILE, axis=1) if confidence is not None else None
    samples = []
    for i, (depth, intrinsic, extrinsic, image_path) in enumerate(zip(depths, intrinsics, normalized, image_paths)):
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError(f"Cannot read RGB frame: {image_path}")
        colour_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dh, dw = depth.shape[:2]; ih, iw = colour_rgb.shape[:2]
        if (ih, iw) != (dh, dw):
            colour_rgb = cv2.resize(colour_rgb, (dw, dh), interpolation=cv2.INTER_LINEAR)
        K = _scale_intrinsics_to_depth(intrinsic, iw, ih, dw, dh)
        valid = np.isfinite(depth) & (depth > 0)
        if confidence is not None:
            if confidence[i].shape != depth.shape:
                raise ValueError("Confidence map shape must match depth map")
            threshold = float(thresholds[i]) if thresholds is not None else -np.inf
            valid &= np.isfinite(confidence[i]) & (confidence[i] >= threshold)
        if not valid.any():
            continue
        v, u = np.nonzero(valid); z = depth[v, u].astype(np.float32)
        points_cam = np.stack([(u.astype(np.float32) - K[0, 2]) * z / K[0, 0], (v.astype(np.float32) - K[1, 2]) * z / K[1, 1], z], axis=1)
        c2w = np.linalg.inv(extrinsic)
        points_world = points_cam @ c2w[:3, :3].T + c2w[:3, 3]
        samples.append(ReconstructionFrameSample(image_path, colour_rgb, depth, K, extrinsic, valid, v, u, points_world * float(scale)))
    return samples


def save_reconstruction_point_cloud(images_dir: Path, depths: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray, output_path: Path, *, confidence: np.ndarray | None = None, scale: float = 1.0) -> Path:
    samples = iter_reconstruction_frame_samples(images_dir, depths, intrinsics, extrinsics, confidence=confidence, scale=scale)
    if not samples:
        raise ValueError("Backprojection produced an empty point cloud")
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.concatenate([s.world_points for s in samples]))
    point_cloud.colors = o3d.utility.Vector3dVector(np.concatenate([s.colour_rgb[s.v_idx, s.u_idx].astype(np.float32) / 255 for s in samples]))
    return _write_point_cloud_output(point_cloud, output_path)


def save_reconstruction_pose_files(frame_timestamps: list[int], extrinsics: np.ndarray, output_dir: Path) -> list[Path]:
    normalized = normalize_extrinsics_to_4x4(extrinsics)
    if normalized is None or len(frame_timestamps) != len(normalized):
        raise ValueError("Frame timestamp count does not match reconstruction extrinsics")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for timestamp, pose in zip(frame_timestamps, normalized):
        path = output_dir / f"{timestamp}.txt"
        path.write_text(" ".join(f"{v:.8f}" for v in pose.flatten()) + "\n")
        paths.append(path)
    return paths


def atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush(); os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)
