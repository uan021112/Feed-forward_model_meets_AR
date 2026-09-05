"""Generic pose math and pose-file helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def rodrigues(omega: np.ndarray) -> np.ndarray:
    """Compute a 3x3 rotation matrix from a rotation vector."""
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.eye(3)
    axis = omega / theta
    skew = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(theta) * skew + (1 - np.cos(theta)) * (skew @ skew)


def rotation_from_gravity(acc: np.ndarray) -> np.ndarray:
    """Build a camera-to-world rotation from the observed gravity direction."""
    acc_norm = np.linalg.norm(acc)
    if acc_norm < 1e-6:
        return np.eye(3)

    z_cam = acc / acc_norm
    reference = np.array([1.0, 0.0, 0.0])
    x_cam = reference - np.dot(reference, z_cam) * z_cam
    if np.linalg.norm(x_cam) < 1e-6:
        reference = np.array([0.0, 1.0, 0.0])
        x_cam = reference - np.dot(reference, z_cam) * z_cam
    x_cam /= np.linalg.norm(x_cam)

    y_cam = np.cross(z_cam, x_cam)
    y_cam /= np.linalg.norm(y_cam)
    return np.column_stack([x_cam, y_cam, z_cam])


def matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a unit quaternion [w, x, y, z]."""
    trace = np.trace(rotation)
    if trace > 0:
        scale = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / scale
        x = (rotation[2, 1] - rotation[1, 2]) * scale
        y = (rotation[0, 2] - rotation[2, 0]) * scale
        z = (rotation[1, 0] - rotation[0, 1]) * scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale
    quaternion = np.array([w, x, y, z])
    return quaternion / np.linalg.norm(quaternion)


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = quaternion
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions: q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions."""
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    w0 = np.sin((1 - t) * theta_0) / sin_theta_0
    w1 = np.sin(t * theta_0) / sin_theta_0
    return w0 * q0 + w1 * q1


def normalize_extrinsics_to_4x4(extrinsics: np.ndarray | None) -> np.ndarray | None:
    """Normalize extrinsics to (N, 4, 4) world-to-camera matrices."""
    if extrinsics is None:
        return None
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics.astype(np.float32)
    if extrinsics.shape[-2:] == (3, 4):
        last_rows = np.tile(
            np.array([0, 0, 0, 1], dtype=np.float32),
            (extrinsics.shape[0], 1, 1),
        )
        return np.concatenate([extrinsics, last_rows], axis=-2).astype(np.float32)
    raise ValueError(f"Unsupported extrinsics shape: {extrinsics.shape}")


def load_pose_extrinsics(
    pose_dir: Path,
    frame_timestamps: list[int],
) -> np.ndarray | None:
    """Load world-to-camera matrices for the requested frame timestamps."""
    pose_paths = [
        pose_dir / f"{timestamp}.txt"
        for timestamp in sorted(frame_timestamps)
        if (pose_dir / f"{timestamp}.txt").exists()
    ]
    return load_pose_matrices(pose_paths)


def load_pose_matrices(pose_paths: Sequence[Path]) -> np.ndarray | None:
    """Load one or more pose text files into stacked 4x4 matrices."""
    world_to_camera = [np.loadtxt(pose_path).reshape(4, 4) for pose_path in pose_paths]
    if not world_to_camera:
        return None
    return np.stack(world_to_camera).astype(np.float32)


def apply_umeyama_scale_to_extrinsics(
    extrinsics: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Scale only the translation component of world-to-camera extrinsics."""
    scaled = normalize_extrinsics_to_4x4(extrinsics)
    if scaled is None:
        raise ValueError("Cannot scale missing extrinsics")
    scaled = scaled.copy()
    scaled[:, :3, 3] *= float(scale)
    return scaled


def camera_centers_from_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    """Convert world-to-camera extrinsics into camera optical centers."""
    normalized = normalize_extrinsics_to_4x4(extrinsics)
    if normalized is None:
        raise ValueError("Cannot compute camera centers from missing extrinsics")
    poses = np.linalg.inv(normalized)
    return poses[:, :3, 3].astype(np.float32)
