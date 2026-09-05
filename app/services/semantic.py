"""LSeg semantic feature extraction for 3D reconstruction.

Loads an LSeg checkpoint, monkey-patches ``extract_features`` onto the ``LSeg``
class, and provides per-frame semantic back-projection to produce a world-space
semantic point cloud saved as NPZ.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import threading
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger(__name__)

# ── Ensure lang-seg is importable ───────────────────────────────────────
_LANG_SEG_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "submodules" / "lang-seg"
)
if str(_LANG_SEG_DIR) not in sys.path:
    sys.path.insert(0, str(_LANG_SEG_DIR))

from app.config import get_lseg_ckpt_path  # noqa: E402
from modules.models.lseg_blocks_zs import forward_vit  # noqa: E402
from modules.models.lseg_net_zs import LSeg, LSegNetZS  # noqa: E402

# ── Monkey-patch LSeg with extract_features ──────────────────────────
def _lseg_extract_features(self: LSeg, x: torch.Tensor) -> torch.Tensor:
    """Extract raw 512-dim CLIP-aligned features from LSeg.

    Runs the ViT backbone + refinenet pipeline, returns
    ``self.scratch.head1(path_1)`` with shape ``(B, 512, H, W)``.
    """
    if self.channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)
    layer_1_rn = self.scratch.layer1_rn(layer_1)
    layer_2_rn = self.scratch.layer2_rn(layer_2)
    layer_3_rn = self.scratch.layer3_rn(layer_3)
    layer_4_rn = self.scratch.layer4_rn(layer_4)
    path_4 = self.scratch.refinenet4(layer_4_rn)
    path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
    path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
    path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
    return self.scratch.head1(path_1)  # (B, 512, H, W)


LSeg.extract_features = _lseg_extract_features


# ── In-memory semantic point cloud cache ─────────────────────────────────
# Avoids the slow NPZ save → load roundtrip for voice queries.
# Keyed by task_id; cleared on explicit unload.

_SEMANTIC_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_SEMANTIC_CACHE_LOCK = threading.Lock()


def cache_semantic_point_cloud(task_id: str, xyz: np.ndarray, features: np.ndarray) -> None:
    """Store semantic point cloud in the in-memory cache."""
    with _SEMANTIC_CACHE_LOCK:
        _SEMANTIC_CACHE[task_id] = (xyz, features)
    logger.debug("Semantic PC cached for task %s (%d pts)", task_id, xyz.shape[0])


def get_cached_semantic_point_cloud(task_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return cached (xyz, features) or None if not present."""
    with _SEMANTIC_CACHE_LOCK:
        return _SEMANTIC_CACHE.get(task_id)


def drop_cached_semantic_point_cloud(task_id: str) -> None:
    """Remove a task's semantic point cloud from the in-memory cache."""
    with _SEMANTIC_CACHE_LOCK:
        if task_id in _SEMANTIC_CACHE:
            del _SEMANTIC_CACHE[task_id]
            logger.debug("Semantic PC cache dropped for task %s", task_id)


def clear_semantic_cache() -> None:
    """Drop all cached semantic point clouds."""
    with _SEMANTIC_CACHE_LOCK:
        count = len(_SEMANTIC_CACHE)
        _SEMANTIC_CACHE.clear()
    if count:
        logger.info("Semantic PC cache cleared (%d entries)", count)
# ── Model cache ─────────────────────────────────────────────────────────

_LSEG_MODEL_LOCK = threading.Lock()
_LSEG_MODEL_CACHE: LSegNetZS | None = None
_LSEG_MODEL_CACHE_DEVICE: str | None = None

CONFIDENCE_PERCENTILE = 10.0
MIN_DEPTH = 0.01
MAX_DEPTH = 100.0


def _read_label_list(dataset: str = "ade20k") -> list[str]:
    """Read LSeg label list from the bundled label_files directory."""
    label_path = _LANG_SEG_DIR / "label_files" / f"fewshot_{dataset}.txt"
    if not label_path.exists():
        raise FileNotFoundError(f"LSeg label file not found: {label_path}")
    with label_path.open() as f:
        return [line.strip() for line in f if line.strip()]


def load_lseg_model(
    ckpt_path: Path | None = None,
    device: str = "cuda:0",
) -> LSegNetZS:
    """Load or reuse the LSeg checkpoint in a thread-safe cache.

    The Lightning checkpoint wraps the model under ``net.*`` keys; these are
    stripped before loading with ``strict=False``.
    """
    global _LSEG_MODEL_CACHE, _LSEG_MODEL_CACHE_DEVICE

    ckpt_path = Path(ckpt_path or get_lseg_ckpt_path())
    with _LSEG_MODEL_LOCK:
        if _LSEG_MODEL_CACHE is not None and _LSEG_MODEL_CACHE_DEVICE == device:
            return _LSEG_MODEL_CACHE

        _clear_lseg_model_cache_locked()

        logger.info("Loading LSeg checkpoint from %s on %s", ckpt_path, device)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        label_list = _read_label_list("ade20k")
        net = LSegNetZS(
            label_list=label_list,
            backbone="clip_vitl16_384",
            features=256,
            aux=False,
            use_pretrained=False,
            arch_option=0,
            block_depth=0,
            activation="lrelu",
        )

        # Strip Lightning "net." prefix
        net_state_dict: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if k.startswith("net."):
                net_state_dict[k[4:]] = v
            else:
                net_state_dict[k] = v

        missing, unexpected = net.load_state_dict(net_state_dict, strict=False)
        if missing:
            logger.warning("LSeg missing keys: %s", missing)
        if unexpected:
            logger.warning("LSeg unexpected keys: %s", unexpected)

        net.eval()
        net.to(device)
        _LSEG_MODEL_CACHE = net
        _LSEG_MODEL_CACHE_DEVICE = device
        logger.info("LSeg model ready on %s", device)
        return net


def _clear_lseg_model_cache_locked() -> None:
    """Release cached LSeg model while holding the cache lock."""
    global _LSEG_MODEL_CACHE, _LSEG_MODEL_CACHE_DEVICE

    model = _LSEG_MODEL_CACHE
    device = _LSEG_MODEL_CACHE_DEVICE or ""
    _LSEG_MODEL_CACHE = None
    _LSEG_MODEL_CACHE_DEVICE = None

    if model is not None:
        if device.startswith("cuda"):
            try:
                model.to("cpu")
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def preload_lseg_model(device: str | None = None) -> str:
    """Warm the shared LSeg model cache and return the resolved device."""
    resolved = device or os.environ.get("LSEG_DEVICE", "cuda:0")
    if resolved.startswith("cuda") and not torch.cuda.is_available():
        resolved = "cpu"
    load_lseg_model(device=resolved)
    return resolved


def unload_lseg_model() -> None:
    """Release the shared LSeg model cache."""
    with _LSEG_MODEL_LOCK:
        _clear_lseg_model_cache_locked()


def get_lseg_model() -> LSegNetZS | None:
    """Return the cached LSeg model, or None if not yet loaded."""
    return _LSEG_MODEL_CACHE


# ── Back-projection ──────────────────────────────────────────────────────

def _w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    """Convert 3×4 world-to-camera extrinsics to 4×4 camera-to-world matrix."""
    R = w2c[:3, :3]   # (3, 3)
    t = w2c[:3, 3]    # (3,)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return c2w


def _back_project_frame(
    depth: np.ndarray,         # (H, W)
    intrinsics: np.ndarray,    # (3, 3)
    extrinsics: np.ndarray,    # (3, 4) world-to-camera
    features: np.ndarray,      # (512, H, W)
    confidence: np.ndarray | None,  # (H, W)
    conf_threshold: float | None = None,  # per-frame confidence percentile threshold
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a single frame's depth+features into world-space points.

    Returns ``(xyz, feats)`` — ``(M, 3)`` and ``(M, 512)`` float32 arrays,
    or empty arrays if no valid pixels.
    """
    H, W = depth.shape

    # Validity mask: depth in range AND confidence above threshold
    mask = (depth > MIN_DEPTH) & (depth < MAX_DEPTH)
    if confidence is not None:
        if conf_threshold is None:
            conf_threshold = float(np.percentile(confidence, CONFIDENCE_PERCENTILE))
        mask &= (confidence > conf_threshold)

    if not mask.any():
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 512), dtype=np.float32)

    # Intrinsic inverse
    K_inv = np.linalg.inv(intrinsics)  # (3, 3)

    # c2w transform
    c2w = _w2c_to_c2w(extrinsics)      # (4, 4)

    # Pixel grid for valid pixels only
    v_idx, u_idx = np.where(mask)
    pixels_h = np.stack([u_idx, v_idx, np.ones_like(u_idx)], axis=1)  # (M, 3)

    # Camera-space points: P_c = depth * K^{-1} @ pixel_h
    cam_pts = (K_inv @ pixels_h.T).T  # (M, 3)
    cam_pts *= depth[v_idx, u_idx, np.newaxis]

    # World-space points: P_w = c2w @ [P_c, 1]
    ones = np.ones((cam_pts.shape[0], 1), dtype=np.float32)
    cam_pts_h = np.concatenate([cam_pts, ones], axis=1)  # (M, 4)
    world_pts = (c2w @ cam_pts_h.T).T[:, :3]             # (M, 3)

    # Gather features
    feats = features[:, v_idx, u_idx].T  # (M, 512)

    return world_pts.astype(np.float32), feats.astype(np.float32)


# ── High-level API ───────────────────────────────────────────────────────

def extract_semantic_point_cloud(
    image_paths: list[Path],
    depth_maps: np.ndarray,       # (N, H, W)
    intrinsics: np.ndarray,       # (N, 3, 3)
    extrinsics: np.ndarray,       # (N, 3, 4) world-to-camera
    confidence: np.ndarray | None,  # (N, H, W) or None
    lseg_net: LSegNetZS,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract semantic features from all frames and back-project into world space.

    Returns ``(xyz, features)``:
    - ``xyz``: (N_total, 3) float32 world-space points in OpenCV RDF frame
    - ``features``: (N_total, 512) float32 L2-normalized per-point features
    """
    N = depth_maps.shape[0]
    H, W = depth_maps.shape[1], depth_maps.shape[2]
    logger.info(
        "Semantic extraction: %d frame(s), resolution %dx%d",
        N, H, W,
    )

    # LSeg requires input padded to multiple of 32
    _align = 32
    H_pad = ((H + _align - 1) // _align) * _align
    W_pad = ((W + _align - 1) // _align) * _align

    lseg_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        T.Resize((H, W), antialias=True),
    ])

    # Compute per-frame confidence thresholds via percentile (3d-lseg convention)
    conf_thresholds: list[float | None] = [None] * len(image_paths)
    if confidence is not None:
        conf_flat = confidence.reshape(confidence.shape[0], -1)
        raw_thresholds: np.ndarray = np.percentile(conf_flat, CONFIDENCE_PERCENTILE, axis=1)
        conf_thresholds = [float(t) for t in raw_thresholds]

    all_xyz: list[np.ndarray] = []
    all_feats: list[np.ndarray] = []
    for i, img_path in enumerate(image_paths):
        if not img_path.exists():
            logger.warning("Frame image missing, skipping: %s", img_path)
            continue

        # Load and transform image
        pil_img = Image.open(img_path).convert("RGB")
        img_tensor = lseg_transform(pil_img).unsqueeze(0).to(device)

        # Pad to multiple of 32
        if H != H_pad or W != W_pad:
            pad_bottom = H_pad - H
            pad_right = W_pad - W
            img_tensor = torch.nn.functional.pad(
                img_tensor, (0, pad_right, 0, pad_bottom), mode="reflect",
            )

        # Extract features
        with torch.no_grad():
            feats = lseg_net.extract_features(img_tensor)  # (1, 512, H_pad/2, W_pad/2)

        # Interpolate back to padded resolution, then crop to original
        feats = torch.nn.functional.interpolate(
            feats, size=(H_pad, W_pad),
            mode="bilinear", align_corners=False,
        )
        feats = feats[:, :, :H, :W]  # (1, 512, H, W)
        feats_np = feats[0].cpu().numpy()  # (512, H, W)

        # Back-project
        depth_i = depth_maps[i]
        K_i = intrinsics[i]
        E_i = extrinsics[i]
        conf_i = confidence[i] if confidence is not None else None
        conf_thr_i = conf_thresholds[i]

        xyz_i, f_i = _back_project_frame(depth_i, K_i, E_i, feats_np, conf_i, conf_thr_i)
        if xyz_i.size == 0:
            logger.debug("Frame %d: no valid semantic points, skipping.", i)
            continue

        all_xyz.append(xyz_i)
        all_feats.append(f_i)

    if not all_xyz:
        logger.warning("No valid semantic points across any frame.")
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 512), dtype=np.float32),
        )

    xyz = np.concatenate(all_xyz, axis=0)
    features = np.concatenate(all_feats, axis=0)

    # L2-normalize features for cosine similarity at query time
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    features = features / norms

    logger.info("Semantic point cloud: %d points", xyz.shape[0])
    return xyz, features


def save_semantic_point_cloud(
    xyz: np.ndarray,
    features: np.ndarray,
    output_path: Path,
) -> Path:
    """Save the semantic point cloud as an uncompressed NPZ file.

    Keys: ``"xyz"`` (float32), ``"features"`` (float32, L2-normalized).
    Uses no compression — the in-memory cache already serves queries;
    this is for crash-recovery persistence only.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        xyz=xyz.astype(np.float32, copy=False),
        features=features.astype(np.float32, copy=False),
    )
    logger.info("Semantic NPZ saved: %s (%d points)", output_path, xyz.shape[0])
    return output_path
