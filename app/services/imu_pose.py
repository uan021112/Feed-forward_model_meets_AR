"""IMU pose export service.

Computes camera orientations via a stateful complementary filter, estimates
translation from gravity-compensated linear acceleration, and saves per-frame
4x4 world2camera matrices.
"""

from pathlib import Path

import numpy as np

from app.utils.imu import parse_imu_csv
from app.utils.pose import (
    matrix_to_quaternion as _matrix_to_quaternion,
    quaternion_multiply as _quaternion_multiply,
    quaternion_to_matrix as _quaternion_to_matrix,
    rodrigues as _rodrigues,
    rotation_from_gravity as _rotation_from_gravity,
    slerp as _slerp,
)

__all__ = [
    "parse_imu_csv",
    "process_imu_orientations",
    "export_pose_files",
    "_rodrigues",
    "_rotation_from_gravity",
    "_matrix_to_quaternion",
    "_quaternion_to_matrix",
    "_quaternion_multiply",
    "_slerp",
    "_interpolate_quaternions",
]


# Complementary filter proportional gain for gravity-direction correction.
# Higher values pull orientation toward accelerometer reference faster.
# Typical range: 1.0–10.0.
KP = 10.0

# Minimum dt (seconds) to guard against divide-by-zero when IMU samples
# have identical timestamps.
MIN_DT = 1e-6

# IMU motion heuristics.
GRAVITY_MPS2 = 9.80665
INVALID_SAMPLE_EPS = 1e-9
STATIONARY_ACCEL_THRESH = 0.5   # m/s² — below this, linear acceleration considered drift/noise
NORMALIZED_G_THRESHOLD = 3.0
VELOCITY_DECAY = 0.95            # per-step velocity decay when accel is below threshold


def _rotate_vector_by_quaternion_inv(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Apply the inverse rotation of quaternion q to vector v (world → camera)."""
    qv = np.array([0.0, v[0], v[1], v[2]])
    q_inv = np.array([q[0], -q[1], -q[2], -q[3]])
    result = _quaternion_multiply(_quaternion_multiply(q_inv, qv), q)
    return result[1:]



def _sanitize_imu_samples(imu_data: list[dict]) -> list[dict]:
    """Drop invalid placeholder rows such as all-zero sensor samples."""
    return [
        row for row in imu_data
        if (
            np.linalg.norm(row["acc"]) >= INVALID_SAMPLE_EPS
            or np.linalg.norm(row["gyro"]) >= INVALID_SAMPLE_EPS
        )
    ]


def _sensor_to_camera_acc(acc_sensor: np.ndarray) -> np.ndarray:
    """Convert accelerometer reading from Android sensor frame to camera frame.

    Android sensor convention (rear camera, portrait):
      Sensor X = camera X (right)
      Sensor Y = up   → camera Y = -sensor Y (camera Y points down)
      Sensor Z = toward user → camera Z = -sensor Z (camera Z points away)
    """
    return np.array([acc_sensor[0], -acc_sensor[1], -acc_sensor[2]], dtype=acc_sensor.dtype)


def _sensor_to_camera_gyro(gyro_sensor: np.ndarray) -> np.ndarray:
    """Convert gyroscope reading from Android sensor frame to camera frame."""
    return np.array([gyro_sensor[0], -gyro_sensor[1], -gyro_sensor[2]], dtype=gyro_sensor.dtype)


def _rotation_from_gravity_y(acc_camera: np.ndarray) -> np.ndarray:
    """Build a camera-to-world rotation from the initial accelerometer reading.

    The normalized accelerometer vector gives the world +Z direction (upward
    force from the hand) expressed in camera coordinates.  This function
    computes a rotation that maps that direction to world [0, 0, 1], with the
    remaining degree of freedom (heading) chosen arbitrarily.

    ``acc_camera`` must already be in camera-frame coordinates.
    Returns a 3x3 rotation matrix (camera → world).
    """
    acc_norm = np.linalg.norm(acc_camera)
    if acc_norm < 1e-6:
        return np.eye(3)

    up_cam = acc_camera / acc_norm  # world +Z direction in camera frame
    up_world = np.array([0.0, 0.0, 1.0])

    # Rotation axis = cross(up_cam, up_world), angle = arccos(dot)
    cos_angle = np.dot(up_cam, up_world)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    if angle < 1e-6:
        return np.eye(3)

    axis = np.cross(up_cam, up_world)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        # up_cam ≈ up_world or ≈ -up_world; identity or 180° rotation
        return np.eye(3) if cos_angle > 0 else -np.eye(3)
    axis /= axis_norm

    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R

def _process_imu_orientations_sanitized(
    imu_data: list[dict],
) -> tuple[list[int], list[np.ndarray]]:
    """Process IMU orientation with a complementary filter.

    The filter expects gravity along the camera Y-axis (down), matching a
    phone held in portrait where the device's vertical axis is camera Y.
    Both accelerometer and gyroscope readings are converted from Android
    sensor frame to camera frame before processing.
    """
    n = len(imu_data)
    if n == 0:
        return [], []

    timestamps = []
    quaternions = []

    # Initial orientation: camera Y aligns with gravity (sensor→camera converted)
    acc_cam0 = _sensor_to_camera_acc(imu_data[0]["acc"])
    R0 = _rotation_from_gravity_y(acc_cam0)
    q = _matrix_to_quaternion(R0)
    timestamps.append(imu_data[0]["timestamp"])
    quaternions.append(q.copy())

    # Gravity reference: world +Z (up) expressed in camera frame.
    # The accelerometer measures the upward force (anti-gravity).
    _UP_WORLD = np.array([0.0, 0.0, 1.0])

    for i in range(1, n):
        dt = (imu_data[i]["timestamp"] - imu_data[i - 1]["timestamp"]) / 1000.0
        dt = max(dt, MIN_DT)

        gyro = _sensor_to_camera_gyro(imu_data[i]["gyro"])
        acc_cam = _sensor_to_camera_acc(imu_data[i]["acc"])
        acc_norm = np.linalg.norm(acc_cam)

        if acc_norm > 1e-6:
            g_meas = acc_cam / acc_norm
            g_est = _rotate_vector_by_quaternion_inv(_UP_WORLD, q)
            error = np.cross(g_meas, g_est)
            gyro = gyro + KP * error
        dt = max(dt, MIN_DT)

        gyro = _sensor_to_camera_gyro(imu_data[i]["gyro"])
        acc_cam = _sensor_to_camera_acc(imu_data[i]["acc"])
        acc_norm = np.linalg.norm(acc_cam)


        R_delta = _rodrigues(gyro * dt)
        q_delta = _matrix_to_quaternion(R_delta)
        q = _quaternion_multiply(q, q_delta)
        q /= np.linalg.norm(q)

        timestamps.append(imu_data[i]["timestamp"])
        quaternions.append(q.copy())

    return timestamps, quaternions



def process_imu_orientations(imu_data: list[dict]) -> tuple[list[int], list[np.ndarray]]:
    """Process IMU data sequentially with a complementary filter.

    Invalid placeholder rows (for example all-zero bootstrap samples) are
    ignored before integration.
    """
    return _process_imu_orientations_sanitized(_sanitize_imu_samples(imu_data))



def _acceleration_to_mps2(acc: np.ndarray, median_acc_norm: float) -> np.ndarray:
    """Normalize accelerometer units to m/s².

    Real uploads often arrive in normalized g-units (median magnitude near 1),
    while synthetic tests and some devices provide m/s² directly.
    """
    scale = GRAVITY_MPS2 if median_acc_norm < NORMALIZED_G_THRESHOLD else 1.0
    return acc * scale



def _integrate_positions(
    imu_data: list[dict],
    imu_quaternions: list[np.ndarray],
) -> list[np.ndarray]:
    """Estimate camera translation from gravity-compensated accelerometer data."""
    if not imu_data:
        return []

    median_acc_norm = float(np.median([np.linalg.norm(row["acc"]) for row in imu_data]))
    gravity_world = np.array([
        0.0,
        0.0,
        float(np.linalg.norm(
            _acceleration_to_mps2(np.array([0.0, 0.0, median_acc_norm]), median_acc_norm)
        )),
    ])
    velocity = np.zeros(3, dtype=np.float64)
    position = np.zeros(3, dtype=np.float64)
    positions = [position.copy()]

    for i in range(1, len(imu_data)):
        dt = (imu_data[i]["timestamp"] - imu_data[i - 1]["timestamp"]) / 1000.0
        dt = max(dt, MIN_DT)

        R_c2w = _quaternion_to_matrix(imu_quaternions[i])
        acc_cam = _sensor_to_camera_acc(imu_data[i]["acc"])
        acc_world = R_c2w @ _acceleration_to_mps2(acc_cam, median_acc_norm)
        linear_acc_world = acc_world - gravity_world

        if np.linalg.norm(linear_acc_world) < STATIONARY_ACCEL_THRESH:
            linear_acc_world = np.zeros(3, dtype=np.float64)
            velocity *= VELOCITY_DECAY

        position = position + velocity * dt + 0.5 * linear_acc_world * dt * dt
        velocity = velocity + linear_acc_world * dt
        positions.append(position.copy())

    return positions

# ---------------------------------------------------------------------------
# Interpolation to frame timestamps
# ---------------------------------------------------------------------------


def _interpolate_quaternions(
    imu_timestamps: list[int],
    imu_quaternions: list[np.ndarray],
    target_timestamps: list[int],
) -> list[np.ndarray]:
    """Slerp-interpolate pre-computed IMU orientations to frame timestamps."""
    ts_arr = np.array(imu_timestamps)
    quats_arr = np.array(imu_quaternions)

    result = []
    for ts in target_timestamps:
        ts = max(ts_arr[0], min(ts_arr[-1], ts))
        idx = np.searchsorted(ts_arr, ts)

        if idx == 0:
            result.append(quats_arr[0].copy())
        elif idx >= len(ts_arr):
            result.append(quats_arr[-1].copy())
        else:
            t0, t1 = ts_arr[idx - 1], ts_arr[idx]
            alpha = float((ts - t0) / (t1 - t0)) if t1 != t0 else 0.0
            result.append(_slerp(quats_arr[idx - 1], quats_arr[idx], alpha))

    return result



def _interpolate_vectors(
    source_timestamps: list[int],
    source_vectors: list[np.ndarray],
    target_timestamps: list[int],
) -> list[np.ndarray]:
    """Linearly interpolate vectors to target timestamps."""
    ts_arr = np.array(source_timestamps)
    vecs_arr = np.array(source_vectors)

    result = []
    for ts in target_timestamps:
        ts = max(ts_arr[0], min(ts_arr[-1], ts))
        idx = np.searchsorted(ts_arr, ts)

        if idx == 0:
            result.append(vecs_arr[0].copy())
        elif idx >= len(ts_arr):
            result.append(vecs_arr[-1].copy())
        else:
            t0, t1 = ts_arr[idx - 1], ts_arr[idx]
            alpha = float((ts - t0) / (t1 - t0)) if t1 != t0 else 0.0
            result.append(vecs_arr[idx - 1] * (1 - alpha) + vecs_arr[idx] * alpha)

    return result


# ---------------------------------------------------------------------------
# Public API: pose export
# ---------------------------------------------------------------------------


def export_pose_files(
    frame_timestamps: list[int],
    imu_data: list[dict],
    output_dir: Path,
) -> list[Path]:
    """Export per-frame pose files as 16-element row-major 4x4 matrices.

    Pipeline:
      1. Recover camera2world orientation from IMU using a complementary filter.
      2. Estimate camera translation by integrating gravity-compensated linear acceleration.
      3. Interpolate orientation/translation to frame timestamps.
      4. Export each frame as a flattened world2camera matrix.
    """
    valid_imu_data = _sanitize_imu_samples(imu_data)
    imu_ts, imu_quats = _process_imu_orientations_sanitized(valid_imu_data)

    if not imu_quats:
        return []

    imu_positions = _integrate_positions(valid_imu_data, imu_quats)
    frame_quats = _interpolate_quaternions(imu_ts, imu_quats, frame_timestamps)
    frame_positions = _interpolate_vectors(imu_ts, imu_positions, frame_timestamps)

    saved = []
    for ts, q, position in zip(frame_timestamps, frame_quats, frame_positions):
        R_c2w = _quaternion_to_matrix(q)
        R_w2c = R_c2w.T
        T = np.eye(4)
        T[:3, :3] = R_w2c
        T[:3, 3] = -R_w2c @ position

        elements = T.flatten().tolist()
        out_path = output_dir / f"{ts}.txt"
        out_path.write_text(" ".join(f"{v:.6f}" for v in elements) + "\n")
        saved.append(out_path)

    return saved
