"""IGGT geometry and instance-feature inference service.

IGGT supplies dense geometry, camera parameters, confidence, and instance
features in one forward pass. LSeg is used by the deployment service to
associate those instances with language-aligned features.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import threading
from types import ModuleType
from contextlib import nullcontext
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from app.config import get_iggt_ckpt_path, get_iggt_device, get_iggt_image_size

logger = logging.getLogger(__name__)

_IGGT_ROOT = Path(__file__).resolve().parents[2] / "submodules" / "IGGT"
_IGGT_MODEL_LOCK = threading.Lock()
_IGGT_MODEL: torch.nn.Module | None = None
_IGGT_MODEL_DEVICE: str | None = None
_IGGT_MODEL_CHECKPOINT: Path | None = None


@dataclass(frozen=True)
class IGGTInferenceResult:
    """Numpy outputs normalized to the backend reconstruction conventions."""

    depths: np.ndarray
    extrinsics: np.ndarray
    intrinsics: np.ndarray
    confidence: np.ndarray
    part_features: np.ndarray
    world_points: np.ndarray


def _compat_module(name: str) -> ModuleType:
    module = sys.modules.setdefault(name, ModuleType(name))
    if module.__spec__ is None:
        module.__spec__ = ModuleSpec(name=name, loader=None)
    return module
def _install_compatibility_modules() -> None:
    """Install tiny optional-dependency shims required by the IGGT model code.

    IGGT imports BasicSR, Detectron2's ``ShapeSpec``, and Apex's RMSNorm even
    though the inference path only needs a small subset of those APIs.
    """
    if "basicsr.archs.arch_util" not in sys.modules:
        basicsr = _compat_module("basicsr")
        archs = _compat_module("basicsr.archs")
        arch_util = _compat_module("basicsr.archs.arch_util")

        def to_2tuple(value: int | tuple[int, int]) -> tuple[int, int]:
            if isinstance(value, tuple):
                if len(value) != 2:
                    raise ValueError(f"Expected a pair, got {value!r}")
                return value
            return value, value

        def trunc_normal_(tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            return torch.nn.init.trunc_normal_(tensor, **kwargs)

        arch_util.to_2tuple = to_2tuple  # type: ignore[attr-defined]
        arch_util.trunc_normal_ = trunc_normal_  # type: ignore[attr-defined]
        sys.modules["basicsr.archs.arch_util"] = arch_util
        setattr(basicsr, "archs", archs)
        setattr(archs, "arch_util", arch_util)

    if "detectron2.layers" not in sys.modules:
        detectron2 = _compat_module("detectron2")
        layers = _compat_module("detectron2.layers")

        class ShapeSpec:
            def __init__(
                self,
                *,
                channels: int | None = None,
                height: int | None = None,
                width: int | None = None,
                stride: int | None = None,
            ) -> None:
                self.channels = channels
                self.height = height
                self.width = width
                self.stride = stride

        layers.ShapeSpec = ShapeSpec  # type: ignore[attr-defined]
        sys.modules["detectron2.layers"] = layers
        setattr(detectron2, "layers", layers)

    if "src.model.norm" not in sys.modules:
        src = _compat_module("src")
        model_pkg = _compat_module("src.model")
        norm = _compat_module("src.model.norm")

        class RMSNorm(torch.nn.Module):
            def __init__(self, dim: int, eps: float | None = None, elementwise_affine: bool = True):
                super().__init__()
                self.eps = eps if eps is not None else torch.finfo(torch.float32).eps
                self.weight = torch.nn.Parameter(torch.ones(dim)) if elementwise_affine else None

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                output = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
                return output * self.weight if self.weight is not None else output

        norm.RMSNorm = RMSNorm  # type: ignore[attr-defined]
        sys.modules["src.model.norm"] = norm
        setattr(src, "model", model_pkg)
        setattr(model_pkg, "norm", norm)


def _ensure_iggt_path() -> None:
    if not _IGGT_ROOT.is_dir():
        raise FileNotFoundError(f"IGGT source directory not found: {_IGGT_ROOT}")
    if str(_IGGT_ROOT) not in sys.path:
        sys.path.insert(0, str(_IGGT_ROOT))
    _install_compatibility_modules()


def _resolve_device(device: str | None = None) -> str:
    requested = device or get_iggt_device()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("IGGT requested %s but CUDA is unavailable; using CPU", requested)
        return "cpu"
    return requested


def _unwrap_checkpoint(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                payload = nested
                break
    if not isinstance(payload, dict) or not all(isinstance(v, torch.Tensor) for v in payload.values()):
        raise ValueError("IGGT checkpoint does not contain a tensor state dictionary")
    return {
        key.removeprefix("module."): value
        for key, value in payload.items()
    }


def _matching_state_dict(
    model: torch.nn.Module,
    checkpoint: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    model_state = model.state_dict()
    matched = {
        key: value
        for key, value in checkpoint.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    if not matched:
        raise ValueError("IGGT checkpoint has no parameters matching the model architecture")
    logger.info(
        "Loaded %d/%d IGGT parameter tensors with matching names and shapes",
        len(matched),
        len(model_state),
    )
    return matched


def _load_model(checkpoint: Path, device: str) -> torch.nn.Module:
    _ensure_iggt_path()
    from iggt.models.vggt import IGGT

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"IGGT checkpoint not found: {checkpoint}. Set IGGT_CKPT_PATH."
        )

    model = IGGT()
    payload = torch.load(checkpoint, map_location=device)
    state_dict = _matching_state_dict(model, _unwrap_checkpoint(payload))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("IGGT checkpoint missing %d model tensors", len(missing))
    if unexpected:
        logger.warning("IGGT checkpoint has %d unused tensors", len(unexpected))
    model.eval().to(device)
    return model


def load_iggt_model(
    device: str | None = None,
    checkpoint: Path | None = None,
) -> torch.nn.Module:
    """Load or reuse the configured IGGT model."""
    global _IGGT_MODEL, _IGGT_MODEL_DEVICE, _IGGT_MODEL_CHECKPOINT

    resolved_device = _resolve_device(device)
    resolved_checkpoint = Path(checkpoint or get_iggt_ckpt_path()).expanduser()
    with _IGGT_MODEL_LOCK:
        if (
            _IGGT_MODEL is not None
            and _IGGT_MODEL_DEVICE == resolved_device
            and _IGGT_MODEL_CHECKPOINT == resolved_checkpoint
        ):
            return _IGGT_MODEL
        _clear_model_cache_locked()
        _IGGT_MODEL = _load_model(resolved_checkpoint, resolved_device)
        _IGGT_MODEL_DEVICE = resolved_device
        _IGGT_MODEL_CHECKPOINT = resolved_checkpoint
        return _IGGT_MODEL


def preload_iggt_model(
    device: str | None = None,
    checkpoint: Path | None = None,
) -> str:
    """Warm the IGGT model cache and return its resolved device."""
    resolved = _resolve_device(device)
    load_iggt_model(resolved, checkpoint)
    return resolved


def _clear_model_cache_locked() -> None:
    global _IGGT_MODEL, _IGGT_MODEL_DEVICE, _IGGT_MODEL_CHECKPOINT
    model = _IGGT_MODEL
    device = _IGGT_MODEL_DEVICE or ""
    _IGGT_MODEL = None
    _IGGT_MODEL_DEVICE = None
    _IGGT_MODEL_CHECKPOINT = None
    if model is not None:
        model.to("cpu")
        del model
        gc.collect()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def unload_iggt_model() -> None:
    """Release the shared IGGT model cache."""
    with _IGGT_MODEL_LOCK:
        _clear_model_cache_locked()


def get_iggt_model() -> torch.nn.Module | None:
    """Return the cached IGGT model, or ``None`` before preload."""
    return _IGGT_MODEL


def _last_pose_encoding(pose_encoding: Any) -> torch.Tensor:
    if isinstance(pose_encoding, (list, tuple)):
        pose_encoding = pose_encoding[-1]
    while isinstance(pose_encoding, torch.Tensor) and pose_encoding.ndim > 3:
        pose_encoding = pose_encoding[-1]
    if not isinstance(pose_encoding, torch.Tensor) or pose_encoding.ndim != 3:
        raise ValueError(f"Unexpected IGGT pose encoding shape: {getattr(pose_encoding, 'shape', None)}")
    return pose_encoding


def _autocast_context(device: str):
    if not device.startswith("cuda"):
        return nullcontext()
    capability = torch.cuda.get_device_capability(torch.device(device))[0]
    dtype = torch.bfloat16 if capability >= 8 else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def run_iggt(
    images_dir: Path,
    *,
    image_paths: Sequence[Path] | None = None,
    device: str | None = None,
    checkpoint: Path | None = None,
) -> IGGTInferenceResult:
    """Run IGGT and return arrays compatible with the existing TSDF pipeline."""
    image_paths = list(image_paths) if image_paths is not None else sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        raise ValueError(f"No JPEG images found in {images_dir}")

    resolved_device = _resolve_device(device)
    model = load_iggt_model(resolved_device, checkpoint)
    _ensure_iggt_path()
    from iggt.utils.load_fn import load_and_preprocess_images
    from iggt.utils.pose_enc import pose_encoding_to_extri_intri

    width, height = get_iggt_image_size()
    images = load_and_preprocess_images(
        image_paths,
        mode="resize",
        resize_target_size=(width, height),
    ).to(resolved_device)

    with torch.inference_mode(), _autocast_context(resolved_device):
        predictions = model(images)

    pose_encoding = _last_pose_encoding(predictions["pose_enc"])
    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        pose_encoding,
        image_size_hw=tuple(images.shape[-2:]),
    )

    def to_numpy(name: str) -> np.ndarray:
        value = predictions.get(name)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"IGGT prediction is missing tensor '{name}'")
        return value.detach().float().cpu().numpy().squeeze(0)

    depths = to_numpy("depth")
    if depths.ndim == 4 and depths.shape[-1] == 1:
        depths = depths[..., 0]
    confidence = to_numpy("depth_conf")
    part_features = to_numpy("part_feat")
    world_points = to_numpy("world_points")
    extrinsics_np = extrinsics.detach().float().cpu().numpy().squeeze(0)
    intrinsics_np = intrinsics.detach().float().cpu().numpy().squeeze(0)

    if depths.ndim != 3 or confidence.shape != depths.shape:
        raise ValueError(
            f"IGGT depth/confidence shapes are incompatible: {depths.shape}, {confidence.shape}"
        )
    if part_features.ndim != 5 or part_features.shape[:2] != (len(image_paths), 1):
        # ``to_numpy`` removes only the batch dimension; normalize the common
        # [S, C, H, W] representation below and reject everything else.
        if part_features.ndim == 4:
            pass
        else:
            raise ValueError(f"Unexpected IGGT part feature shape: {part_features.shape}")
    if part_features.ndim == 5:
        part_features = part_features[:, 0]
    if world_points.ndim == 5:
        world_points = world_points[:, 0]

    if len(depths) != len(image_paths) or extrinsics_np.shape[0] != len(image_paths):
        raise ValueError("IGGT output frame count does not match extracted images")
    if part_features.ndim != 4 or world_points.shape[:3] != depths.shape:
        raise ValueError(
            f"Unexpected IGGT feature/world-point shapes: {part_features.shape}, {world_points.shape}"
        )

    return IGGTInferenceResult(
        depths=depths.astype(np.float32, copy=False),
        extrinsics=extrinsics_np.astype(np.float32, copy=False),
        intrinsics=intrinsics_np.astype(np.float32, copy=False),
        confidence=confidence.astype(np.float32, copy=False),
        part_features=part_features.astype(np.float32, copy=False),
        world_points=world_points.astype(np.float32, copy=False),
    )


def save_iggt_instance_point_cloud(
    result: IGGTInferenceResult,
    output_path: Path,
) -> Path:
    """Persist normalized IGGT instance features and geometric confidence."""
    xyz = result.world_points.reshape(-1, 3)
    features = result.part_features.transpose(0, 2, 3, 1).reshape(-1, result.part_features.shape[1])
    confidence = result.confidence.reshape(-1)
    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(features).all(axis=1) & np.isfinite(confidence) & (confidence > 0)
    xyz, features, confidence = xyz[valid], features[valid], confidence[valid]
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, format_version=np.asarray([1]), xyz=xyz.astype(np.float32), features=features.astype(np.float32), confidence=confidence.astype(np.float32))
    logger.info("IGGT instance point cloud saved: %s (%d points)", output_path, xyz.shape[0])
    return output_path
