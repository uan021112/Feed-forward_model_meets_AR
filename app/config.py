"""Configuration for the IGGT mobile AR deployment service."""

from __future__ import annotations

import os
from pathlib import Path


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(int(os.environ.get(name, str(default))), minimum)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def get_reconstruct_base_dir() -> Path:
    return Path(os.environ.get("RECONSTRUCT_BASE_DIR", "/home/tcluan/D-Data/ExpLog/icxr-backend/reconstruct_tasks"))


def get_frame_extract_fps() -> float:
    return _env_float("FRAME_EXTRACT_FPS", 1.0, 0.01)


def get_lseg_ckpt_path() -> Path:
    return Path(os.environ.get("LSEG_CKPT_PATH", "/mnt/data/tcluan/Ckpt/lseg/demo_e200.ckpt"))


def get_iggt_ckpt_path() -> Path:
    return Path(os.environ.get("IGGT_CKPT_PATH", "/mnt/data/tcluan/Ckpt/iggt/IGGT_official.pth"))


def get_iggt_device() -> str:
    return os.environ.get("IGGT_DEVICE", "cuda:0")


def get_iggt_image_size() -> tuple[int, int]:
    raw = os.environ.get("IGGT_IMAGE_SIZE", "392x630").lower().replace(",", "x")
    try:
        width, height = (int(part.strip()) for part in raw.split("x"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid IGGT_IMAGE_SIZE={raw!r}; expected WIDTHxHEIGHT") from exc
    if width <= 0 or height <= 0:
        raise ValueError("IGGT_IMAGE_SIZE must be positive")
    return width, height


def get_reference_max_views() -> int:
    return _env_int("IGGT_MAX_REFERENCE_VIEWS", 50, 3)


def get_reference_retrieval_count() -> int:
    return _env_int("IGGT_REFERENCE_RETRIEVAL_COUNT", 10, 1)


def get_reference_min_confidence() -> float:
    return min(max(_env_float("IGGT_RELOCALIZATION_MIN_CONFIDENCE", 0.5), 0.0), 1.0)


def get_reference_reliability_weight() -> float:
    return _env_float("IGGT_REFERENCE_RELIABILITY_WEIGHT", 0.5, 0.0)


def get_reference_novelty_weight() -> float:
    return _env_float("IGGT_REFERENCE_NOVELTY_WEIGHT", 0.5, 0.0)


def get_reference_min_connections() -> int:
    return _env_int("IGGT_REFERENCE_MIN_CONNECTIONS", 2, 1)


def get_reference_connection_threshold() -> float:
    return min(max(_env_float("IGGT_REFERENCE_CONNECTION_THRESHOLD", 0.2), 0.0), 1.0)


def get_reference_reward_margin() -> float:
    return _env_float("IGGT_REFERENCE_REWARD_MARGIN", 0.05, 0.0)


def get_deployment_alignment_max_rms() -> float:
    return _env_float("IGGT_DEPLOYMENT_MAX_ALIGNMENT_RMS", 0.05, 0.0)


def get_runtime_alignment_max_rms() -> float:
    return _env_float("IGGT_RUNTIME_MAX_ALIGNMENT_RMS", 0.05, 0.0)


def get_semantic_association_max_distance() -> float:
    return _env_float("IGGT_LSEG_ASSOCIATION_MAX_DISTANCE", 0.05, 0.0)


def get_mesh_association_max_distance() -> float:
    return _env_float("IGGT_MESH_ASSOCIATION_MAX_DISTANCE", 0.05, 0.0)


def get_min_instance_points() -> int:
    return _env_int("IGGT_MIN_INSTANCE_POINTS", 50, 1)


def get_min_lseg_associations() -> int:
    return _env_int("IGGT_MIN_LSEG_ASSOCIATIONS", 10, 1)


def get_instance_cluster_eps() -> float:
    return _env_float("IGGT_INSTANCE_CLUSTER_EPS", 0.05, 0.0)


def get_instance_cluster_min_samples() -> int:
    return _env_int("IGGT_INSTANCE_CLUSTER_MIN_SAMPLES", 10, 1)


def get_semantic_query_min_similarity() -> float:
    return _env_float("IGGT_SEMANTIC_QUERY_MIN_SIMILARITY", 0.25)


def get_runtime_interval_seconds() -> float:
    return _env_float("IGGT_RELOCALIZATION_INTERVAL_SECONDS", 10.0, 0.1)
