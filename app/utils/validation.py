"""Generic payload validation helpers."""

from __future__ import annotations

import csv
import io


_AR_POSE_REQUIRED_COLS = {"timestamp", "px", "py", "pz", "qx", "qy", "qz", "qw"}


def validate_ar_pose_csv(content: bytes) -> None:
    """Validate that uploaded AR pose content is a non-empty 8-column CSV
    with columns: timestamp, px, py, pz, qx, qy, qz, qw."""
    text = content.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("AR pose CSV file is empty")

    if not _AR_POSE_REQUIRED_COLS.issubset(set(reader.fieldnames)):
        raise ValueError(
            f"AR pose CSV must have 8 columns: {_AR_POSE_REQUIRED_COLS}. "
            f"Found: {reader.fieldnames}"
        )
