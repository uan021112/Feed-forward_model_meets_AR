"""Tests for IMU pose calculation service.

Uses synthetic IMU data derived from known ground-truth orientation
trajectories to verify correctness of parsing, complementary filter,
quaternion utilities, and pose file export.
"""

import csv
import io
import math
from pathlib import Path

import numpy as np
import pytest

from app.services.imu_pose import (
    parse_imu_csv,
    process_imu_orientations,
    export_pose_files,
    _rodrigues,
    _rotation_from_gravity,
    _matrix_to_quaternion,
    _quaternion_to_matrix,
    _slerp,
    _quaternion_multiply,
    _interpolate_quaternions,
)

G = 9.81  # gravitational acceleration (m/s²)


# ===========================================================================
# Synthetic IMU data generators
# ===========================================================================


def _make_csv_content(rows: list[dict]) -> str:
    """Serialize rows into a 7-column IMU CSV string."""
    header = "timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z"
    lines = [header]
    for r in rows:
        lines.append(
            f"{r['timestamp']},"
            f"{r['acc'][0]:.6f},{r['acc'][1]:.6f},{r['acc'][2]:.6f},"
            f"{r['gyro'][0]:.6f},{r['gyro'][1]:.6f},{r['gyro'][2]:.6f}"
        )
    return "\n".join(lines) + "\n"


def _write_csv(tmp_path: Path, filename: str, rows: list[dict]) -> Path:
    """Write synthetic IMU rows to a temporary CSV file."""
    p = tmp_path / filename
    p.write_text(_make_csv_content(rows))
    return p


def generate_static_imu(duration_ms: int = 1000, interval_ms: int = 10) -> list[dict]:
    """Generate IMU data for a stationary upright camera.

    Accelerometer reads [0, 0, G]; gyro reads [0, 0, 0].
    """
    rows = []
    for t in range(0, duration_ms + interval_ms, interval_ms):
        rows.append({
            "timestamp": t,
            "acc": np.array([0.0, 0.0, G]),
            "gyro": np.array([0.0, 0.0, 0.0]),
        })
    return rows


def generate_pitch_rotation(
    total_angle_rad: float = math.pi / 2,
    duration_ms: int = 1000,
    interval_ms: int = 10,
) -> list[dict]:
    """Generate IMU data for a constant-rate pitch rotation (around X axis).

    The camera pitches forward at constant angular velocity.  Synthetic
    accel/gyro values are derived from the ground-truth orientation.

    Args:
        total_angle_rad: Total pitch angle in radians.
        duration_ms: Duration in milliseconds.
        interval_ms: IMU sample interval in milliseconds.

    Returns:
        List of IMU sample dicts with ground-truth timestamps.
    """
    rows = []
    num_samples = duration_ms // interval_ms + 1
    omega = total_angle_rad / (duration_ms / 1000.0)  # rad/s

    for i in range(num_samples):
        t_ms = i * interval_ms
        theta = omega * (t_ms / 1000.0)  # current pitch angle

        # Ground-truth orientation: rotation around X by theta
        c, s = math.cos(theta), math.sin(theta)
        R_gt = np.array([
            [1.0, 0.0, 0.0],
            [0.0, c,   -s],
            [0.0, s,   c],
        ])

        # Accelerometer: R^T @ gravity_world (Z-down world frame, gravity = [0,0,G])
        acc = R_gt.T @ np.array([0.0, 0.0, G])

        # Gyroscope: angular velocity in camera frame (constant around X)
        gyro = np.array([omega, 0.0, 0.0])

        rows.append({
            "timestamp": t_ms,
            "acc": acc,
            "gyro": gyro,
        })
    return rows


def generate_roll_rotation(
    total_angle_rad: float = math.pi / 2,
    duration_ms: int = 1000,
    interval_ms: int = 10,
) -> list[dict]:
    """Generate IMU data for a constant-rate roll rotation (around Y axis).

    The camera rolls sideways at constant angular velocity.
    """
    rows = []
    num_samples = duration_ms // interval_ms + 1
    omega = total_angle_rad / (duration_ms / 1000.0)

    for i in range(num_samples):
        t_ms = i * interval_ms
        theta = omega * (t_ms / 1000.0)

        c, s = math.cos(theta), math.sin(theta)
        R_gt = np.array([
            [c,  0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ])

        acc = R_gt.T @ np.array([0.0, 0.0, G])
        gyro = np.array([0.0, omega, 0.0])

        rows.append({
            "timestamp": t_ms,
            "acc": acc,
            "gyro": gyro,
        })
    return rows


def generate_yaw_rotation(
    total_angle_rad: float = math.pi / 2,
    duration_ms: int = 1000,
    interval_ms: int = 10,
) -> list[dict]:
    """Generate IMU data for a constant-rate yaw rotation (around Z axis).

    Note: yaw is unobservable from accelerometer — gravity reading remains
    [0, 0, G] throughout.  Only gyro integration can track yaw.
    """
    rows = []
    num_samples = duration_ms // interval_ms + 1
    omega = total_angle_rad / (duration_ms / 1000.0)

    for i in range(num_samples):
        t_ms = i * interval_ms
        theta = omega * (t_ms / 1000.0)

        c, s = math.cos(theta), math.sin(theta)
        R_gt = np.array([
            [c,   -s, 0.0],
            [s,   c,   0.0],
            [0.0, 0.0, 1.0],
        ])

        acc = R_gt.T @ np.array([0.0, 0.0, G])
        gyro = np.array([0.0, 0.0, omega])

        rows.append({
            "timestamp": t_ms,
            "acc": acc,
            "gyro": gyro,
        })
    return rows


def generate_linear_translation(
    acceleration_mps2: float = 1.0,
    duration_ms: int = 1000,
    interval_ms: int = 10,
) -> list[dict]:
    """Generate upright motion with constant world-X acceleration."""
    rows = []
    for t in range(0, duration_ms + interval_ms, interval_ms):
        rows.append({
            "timestamp": t,
            "acc": np.array([acceleration_mps2, 0.0, G]),
            "gyro": np.array([0.0, 0.0, 0.0]),
        })
    return rows


# ===========================================================================
# US-002: parse_imu_csv tests
# ===========================================================================


class TestParseImuCsv:
    def test_parse_valid_csv(self, tmp_path):
        rows = generate_static_imu(duration_ms=50, interval_ms=10)
        p = _write_csv(tmp_path, "valid.csv", rows)

        result = parse_imu_csv(p)
        assert len(result) == len(rows)
        assert all("timestamp" in r for r in result)
        assert all("acc" in r for r in result)
        assert all("gyro" in r for r in result)

    def test_parse_empty_file(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("")

        with pytest.raises(ValueError, match="empty"):
            parse_imu_csv(p)

    def test_parse_missing_columns(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("timestamp,acc_x,acc_y\n1,0,0\n")

        with pytest.raises(ValueError, match="7 columns"):
            parse_imu_csv(p)

    def test_parse_no_data_rows(self, tmp_path):
        p = tmp_path / "header_only.csv"
        p.write_text("timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z\n")

        with pytest.raises(ValueError, match="no data rows"):
            parse_imu_csv(p)

    def test_parse_single_row(self, tmp_path):
        rows = [{"timestamp": 0, "acc": np.array([0., 0., G]), "gyro": np.array([0., 0., 0.])}]
        p = _write_csv(tmp_path, "single.csv", rows)

        result = parse_imu_csv(p)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0]["acc"], rows[0]["acc"])

    def test_parse_sorts_by_timestamp(self, tmp_path):
        rows = [
            {"timestamp": 20, "acc": np.array([0., 0., G]), "gyro": np.array([0., 0., 0.])},
            {"timestamp": 0,  "acc": np.array([0., 0., G]), "gyro": np.array([0., 0., 0.])},
            {"timestamp": 10, "acc": np.array([0., 0., G]), "gyro": np.array([0., 0., 0.])},
        ]
        p = _write_csv(tmp_path, "unsorted.csv", rows)

        result = parse_imu_csv(p)
        timestamps = [r["timestamp"] for r in result]
        assert timestamps == [0, 10, 20]


# ===========================================================================
# US-003: Rotation / quaternion utility tests
# ===========================================================================


class TestRodrigues:
    def test_zero_rotation_returns_identity(self):
        R = _rodrigues(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_90deg_around_z(self):
        R = _rodrigues(np.array([0.0, 0.0, math.pi / 2]))
        expected = np.array([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0],
        ])
        np.testing.assert_array_almost_equal(R, expected)

    def test_90deg_around_x(self):
        R = _rodrigues(np.array([math.pi / 2, 0.0, 0.0]))
        expected = np.array([
            [1.0, 0.0,  0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0,  0.0],
        ])
        np.testing.assert_array_almost_equal(R, expected)


class TestRotationFromGravity:
    def test_upright_camera(self):
        R = _rotation_from_gravity(np.array([0.0, 0.0, G]))
        # Should be identity — Z aligns with gravity measurement
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_z_column_is_unit_gravity_direction(self):
        acc = np.array([1.0, 2.0, 3.0])
        R = _rotation_from_gravity(acc)
        expected_z = acc / np.linalg.norm(acc)
        np.testing.assert_array_almost_equal(R[:, 2], expected_z)

    def test_matrix_is_orthonormal(self):
        acc = np.array([0.5, -0.3, 8.0])
        R = _rotation_from_gravity(acc)
        np.testing.assert_array_almost_equal(R.T @ R, np.eye(3), decimal=6)
        assert np.linalg.det(R) > 0  # right-handed

    def test_zero_accel_returns_identity(self):
        R = _rotation_from_gravity(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_gravity_along_x_axis(self):
        acc = np.array([G, 0.0, 0.0])
        R = _rotation_from_gravity(acc)
        np.testing.assert_array_almost_equal(R[:, 2], np.array([1.0, 0.0, 0.0]))


class TestQuaternionRoundTrip:
    def test_identity(self):
        R = np.eye(3)
        q = _matrix_to_quaternion(R)
        R_back = _quaternion_to_matrix(q)
        np.testing.assert_array_almost_equal(R, R_back)

    def test_rotation_x_90(self):
        R = _rodrigues(np.array([math.pi / 2, 0.0, 0.0]))
        q = _matrix_to_quaternion(R)
        R_back = _quaternion_to_matrix(q)
        np.testing.assert_array_almost_equal(R, R_back)

    def test_rotation_y_45(self):
        R = _rodrigues(np.array([0.0, math.pi / 4, 0.0]))
        q = _matrix_to_quaternion(R)
        R_back = _quaternion_to_matrix(q)
        np.testing.assert_array_almost_equal(R, R_back)

    def test_rotation_z_30(self):
        R = _rodrigues(np.array([0.0, 0.0, math.pi / 6]))
        q = _matrix_to_quaternion(R)
        R_back = _quaternion_to_matrix(q)
        np.testing.assert_array_almost_equal(R, R_back)

    def test_arbitrary_rotation(self):
        omega = np.array([0.3, -0.5, 0.7])
        R = _rodrigues(omega)
        q = _matrix_to_quaternion(R)
        R_back = _quaternion_to_matrix(q)
        np.testing.assert_array_almost_equal(R, R_back)


class TestSlerp:
    def test_slerp_t0_returns_q0(self):
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        q1 = np.array([0.0, 1.0, 0.0, 0.0])
        result = _slerp(q0, q1, 0.0)
        np.testing.assert_array_almost_equal(result, q0)

    def test_slerp_t1_returns_q1(self):
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        q1 = np.array([0.0, 1.0, 0.0, 0.0])
        result = _slerp(q0, q1, 1.0)
        np.testing.assert_array_almost_equal(result, q1)

    def test_slerp_midpoint(self):
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        q1 = np.array([0.0, 1.0, 0.0, 0.0])
        result = _slerp(q0, q1, 0.5)
        expected = np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_slerp_shortest_path(self):
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        q1 = np.array([0.0, -1.0, 0.0, 0.0])
        q1_neg = np.array([0.0, 1.0, 0.0, 0.0])
        # Both should give same result since q and -q represent the same rotation
        result1 = _slerp(q0, q1, 0.5)
        result2 = _slerp(q0, q1_neg, 0.5)
        np.testing.assert_array_almost_equal(np.abs(result1), np.abs(result2))


class TestQuaternionMultiply:
    def test_identity_multiply(self):
        q = np.array([0.70710678, 0.70710678, 0.0, 0.0])  # 90 deg around X
        q_identity = np.array([1.0, 0.0, 0.0, 0.0])
        result = _quaternion_multiply(q, q_identity)
        np.testing.assert_array_almost_equal(result, q)


# ===========================================================================
# US-004: process_imu_orientations (stateful complementary filter) tests
# ===========================================================================


class TestProcessImuOrientations:
    def test_empty_data_returns_empty(self):
        ts, quats = process_imu_orientations([])
        assert ts == []
        assert quats == []

    def test_ignores_leading_all_zero_placeholder_sample(self):
        data = [{
            "timestamp": 0,
            "acc": np.array([0.0, 0.0, 0.0]),
            "gyro": np.array([0.0, 0.0, 0.0]),
        }] + [
            {
                "timestamp": row["timestamp"] + 10,
                "acc": row["acc"],
                "gyro": row["gyro"],
            }
            for row in generate_static_imu(duration_ms=20, interval_ms=10)
        ]

        ts, quats = process_imu_orientations(data)

        assert ts[0] == 10
        np.testing.assert_array_almost_equal(quats[0], np.array([1.0, 0.0, 0.0, 0.0]))

    def test_single_sample(self):
        data = [{"timestamp": 0, "acc": np.array([0., 0., G]), "gyro": np.array([0., 0., 0.])}]
        ts, quats = process_imu_orientations(data)
        assert len(ts) == 1
        assert len(quats) == 1
        # Should be identity quaternion from upright gravity
        np.testing.assert_array_almost_equal(quats[0], np.array([1.0, 0.0, 0.0, 0.0]), decimal=4)

    def test_static_camera_stays_near_identity(self):
        """Static camera: all orientations should stay near identity."""
        data = generate_static_imu(duration_ms=1000, interval_ms=10)
        ts, quats = process_imu_orientations(data)

        for q in quats:
            # Identity quaternion has w ≈ 1, xyz ≈ 0
            assert abs(q[0] - 1.0) < 0.01
            assert np.linalg.norm(q[1:]) < 0.01
            assert abs(np.linalg.norm(q) - 1.0) < 1e-10

    def test_pitch_rotation_accumulates(self):
        """Pitch rotation: first and last orientations must differ."""
        data = generate_pitch_rotation(
            total_angle_rad=math.pi / 2, duration_ms=1000, interval_ms=10
        )
        ts, quats = process_imu_orientations(data)

        # First quaternion is from gravity (near identity)
        # Last quaternion should encode accumulated ~90° pitch rotation
        q_first = quats[0]
        q_last = quats[-1]

        # They should NOT be the same
        assert np.linalg.norm(q_last - q_first) > 0.1

    def test_pitch_rotation_axis_is_x(self):
        """Pitch rotation should primarily rotate around the X axis."""
        data = generate_pitch_rotation(
            total_angle_rad=math.pi / 2, duration_ms=1000, interval_ms=10
        )
        ts, quats = process_imu_orientations(data)

        q_last = quats[-1]
        # For rotation around X: q = [cos(θ/2), sin(θ/2), 0, 0]
        # The Y and Z components should be small compared to X
        assert abs(q_last[1]) > abs(q_last[2]) * 0.5
        assert abs(q_last[1]) > abs(q_last[3]) * 0.5

    def test_roll_rotation_accumulates(self):
        """Roll rotation: orientations must change over time."""
        data = generate_roll_rotation(
            total_angle_rad=math.pi / 2, duration_ms=1000, interval_ms=10
        )
        ts, quats = process_imu_orientations(data)

        q_first = quats[0]
        q_last = quats[-1]
        assert np.linalg.norm(q_last - q_first) > 0.1

    def test_yaw_rotation_accumulates(self):
        """Yaw rotation: tracked by gyro integration alone (accel provides no yaw)."""
        data = generate_yaw_rotation(
            total_angle_rad=math.pi / 2, duration_ms=1000, interval_ms=10
        )
        ts, quats = process_imu_orientations(data)

        q_first = quats[0]
        q_last = quats[-1]
        assert np.linalg.norm(q_last - q_first) > 0.1

    def test_pitch_vs_ground_truth(self):
        """For a 90° pitch, the recovered pitch angle should be within 15% of ground truth.

        We extract the pitch angle from the final quaternion and compare.
        """
        data = generate_pitch_rotation(
            total_angle_rad=math.pi / 2, duration_ms=1000, interval_ms=10
        )
        ts, quats = process_imu_orientations(data)

        # Extract pitch from the final quaternion
        # For rotation around X: q = [cos(θ/2), sin(θ/2), 0, 0]
        q_last = quats[-1]
        # Recover the angle from the X rotation component
        # pitch angle = 2 * asin(x_component), but with complementary filter
        # there may be small Y/Z components, so use: θ ≈ 2 * atan2(|xyz|, w)
        recovered_angle = 2.0 * math.atan2(
            np.linalg.norm(q_last[1:]), abs(q_last[0])
        )
        expected_angle = math.pi / 2

        relative_error = abs(recovered_angle - expected_angle) / expected_angle
        assert relative_error < 0.15, (
            f"Pitch recovery error {relative_error:.2%} exceeds 15%. "
            f"Expected {expected_angle:.3f} rad, got {recovered_angle:.3f} rad"
        )

    def test_all_quaternions_are_unit_norm(self):
        """Every output quaternion must have unit norm."""
        data = generate_pitch_rotation(
            total_angle_rad=math.pi / 2, duration_ms=500, interval_ms=10
        )
        ts, quats = process_imu_orientations(data)

        for q in quats:
            assert abs(np.linalg.norm(q) - 1.0) < 1e-10


# ===========================================================================
# US-005: _interpolate_quaternions and export_pose_files integration tests
# ===========================================================================


class TestInterpolateQuaternions:
    def test_interpolate_at_exact_timestamps(self):
        ts = [0, 100, 200]
        quats = [
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 0.0]),
        ]
        result = _interpolate_quaternions(ts, quats, [0, 100, 200])
        np.testing.assert_array_almost_equal(result[0], quats[0])
        np.testing.assert_array_almost_equal(result[1], quats[1])
        np.testing.assert_array_almost_equal(result[2], quats[2])

    def test_interpolate_between_samples(self):
        ts = [0, 100]
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        q1 = np.array([0.0, 1.0, 0.0, 0.0])
        result = _interpolate_quaternions(ts, [q0, q1], [50])

        # Midpoint slerp between identity and 180° rotation around X
        expected = _slerp(q0, q1, 0.5)
        np.testing.assert_array_almost_equal(result[0], expected)

    def test_interpolate_clamps_to_bounds(self):
        ts = [100, 200]
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        q1 = np.array([0.0, 1.0, 0.0, 0.0])
        result = _interpolate_quaternions(ts, [q0, q1], [0, 300])

        # Before first → clamped to first
        np.testing.assert_array_almost_equal(result[0], q0)
        # After last → clamped to last
        np.testing.assert_array_almost_equal(result[1], q1)


class TestExportPoseFiles:
    def test_creates_correct_number_of_files(self, tmp_path):
        data = generate_static_imu(duration_ms=100, interval_ms=10)
        frame_ts = [0, 50, 100]
        out_dir = tmp_path / "pose"
        out_dir.mkdir()

        saved = export_pose_files(frame_ts, data, out_dir)
        assert len(saved) == 3
        for p in saved:
            assert p.exists()

    def test_file_names_are_timestamps(self, tmp_path):
        data = generate_static_imu(duration_ms=100, interval_ms=10)
        frame_ts = [12345, 67890]
        out_dir = tmp_path / "pose"
        out_dir.mkdir()

        saved = export_pose_files(frame_ts, data, out_dir)
        assert saved[0].name == "12345.txt"
        assert saved[1].name == "67890.txt"

    def test_pose_file_format(self, tmp_path):
        """Each pose file should have exactly 16 whitespace-separated numbers."""
        data = generate_static_imu(duration_ms=100, interval_ms=10)
        frame_ts = [50]
        out_dir = tmp_path / "pose"
        out_dir.mkdir()

        saved = export_pose_files(frame_ts, data, out_dir)
        content = saved[0].read_text().strip()
        parts = content.split()

        assert len(parts) == 16
        for p in parts:
            # Each number should have 6 decimal places
            int_part, frac_part = p.split(".")
            assert len(frac_part) == 6

    def test_pose_matrix_is_valid_transformation(self, tmp_path):
        """The 4x4 matrix must: (a) have rotation part with det=±1,
        (b) have bottom row [0,0,0,1], (c) have zero translation."""
        data = generate_static_imu(duration_ms=100, interval_ms=10)
        frame_ts = [0, 50, 100]
        out_dir = tmp_path / "pose"
        out_dir.mkdir()

        saved = export_pose_files(frame_ts, data, out_dir)
        for p in saved:
            nums = [float(x) for x in p.read_text().strip().split()]
            T = np.array(nums).reshape(4, 4)

            # Bottom row
            np.testing.assert_array_almost_equal(T[3], np.array([0., 0., 0., 1.]))

            # Rotation part determinant ≈ 1
            R = T[:3, :3]
            assert abs(np.linalg.det(R) - 1.0) < 1e-6

            # Translation is zero
            np.testing.assert_array_almost_equal(T[:3, 3], np.array([0., 0., 0.]))

    def test_pose_matrix_uses_world2camera_rotation(self, tmp_path):
        total_angle = math.pi / 4
        data = generate_pitch_rotation(
            total_angle_rad=total_angle, duration_ms=1000, interval_ms=10
        )
        out_dir = tmp_path / "pose"
        out_dir.mkdir()

        saved = export_pose_files([1000], data, out_dir)
        nums = [float(x) for x in saved[0].read_text().strip().split()]
        T = np.array(nums).reshape(4, 4)

        c, s = math.cos(total_angle), math.sin(total_angle)
        expected_w2c = np.array([
            [1.0, 0.0, 0.0],
            [0.0, c, s],
            [0.0, -s, c],
        ])
        np.testing.assert_array_almost_equal(T[:3, :3], expected_w2c, decimal=2)

    def test_linear_motion_produces_nonzero_translation(self, tmp_path):
        data = generate_linear_translation(
            acceleration_mps2=1.0, duration_ms=1000, interval_ms=10
        )
        frame_ts = [0, 500, 1000]
        out_dir = tmp_path / "pose"
        out_dir.mkdir()

        saved = export_pose_files(frame_ts, data, out_dir)
        matrices = []
        for p in saved:
            nums = [float(x) for x in p.read_text().strip().split()]
            matrices.append(np.array(nums).reshape(4, 4))

        np.testing.assert_array_almost_equal(matrices[0][:3, 3], np.array([0.0, 0.0, 0.0]))
        assert matrices[-1][0, 3] < -0.01
        assert abs(matrices[-1][1, 3]) < 0.02
        assert abs(matrices[-1][2, 3]) < 0.02

    def test_full_pipeline_with_pitch(self, tmp_path):
        """End-to-end: synthetic pitch data → pose files with rotating matrix."""
        data = generate_pitch_rotation(
            total_angle_rad=math.pi / 4, duration_ms=1000, interval_ms=10
        )
        frame_ts = [0, 500, 1000]
        out_dir = tmp_path / "pose"
        out_dir.mkdir()

        saved = export_pose_files(frame_ts, data, out_dir)
        assert len(saved) == 3

        # Read the matrices
        matrices = []
        for p in saved:
            nums = [float(x) for x in p.read_text().strip().split()]
            matrices.append(np.array(nums).reshape(4, 4))

        # First frame (t=0) should be near identity
        np.testing.assert_array_almost_equal(matrices[0][:3, :3], np.eye(3), decimal=3)

        # Last frame (t=1000, 45° pitch) should NOT be identity
        diff = np.linalg.norm(matrices[-1][:3, :3] - np.eye(3))
        assert diff > 0.1, "Expected non-identity rotation after 45° pitch"

    def test_empty_imu_data_graceful(self, tmp_path):
        out_dir = tmp_path / "pose"
        out_dir.mkdir()
        saved = export_pose_files([100], [], out_dir)
        assert saved == []
