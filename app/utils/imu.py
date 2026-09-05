"""Generic IMU parsing helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def parse_imu_csv(csv_path: Path) -> list[dict]:
    """Parse and validate a 7-column IMU CSV file."""
    rows = []
    with open(csv_path, "r", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("IMU CSV file is empty")

        required = {"timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"}
        if not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"IMU CSV must have 7 columns: {required}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            rows.append({
                "timestamp": int(row["timestamp"]),
                "acc": np.array([
                    float(row["acc_x"]),
                    float(row["acc_y"]),
                    float(row["acc_z"]),
                ]),
                "gyro": np.array([
                    float(row["gyro_x"]),
                    float(row["gyro_y"]),
                    float(row["gyro_z"]),
                ]),
            })

    if not rows:
        raise ValueError("IMU CSV file contains no data rows")

    rows.sort(key=lambda row: row["timestamp"])
    return rows
