"""TSDF volumetric fusion and mesh export service.

Fuses per-frame depth maps into a truncated signed-distance field using Open3D,
then extracts the surface mesh and exports it as a GLB file.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
import numpy as np
import open3d as o3d
import trimesh


VOXEL_LENGTH = 0.01
SDF_TRUNC = 0.04
MAX_DEPTH = 50.0
BLOCK_RESOLUTION = 16
BLOCK_COUNT = 50_000
MESH_WEIGHT_THRESHOLD = 1.0

def _get_tsdf_device() -> o3d.core.Device:
    """Prefer CUDA for tensor TSDF fusion when Open3D exposes it."""
    if hasattr(o3d.core, "cuda") and o3d.core.cuda.is_available():
        return o3d.core.Device("CUDA:0")
    return o3d.core.Device("CPU:0")


def _create_voxel_block_grid(
    *,
    voxel_length: float,
    device: o3d.core.Device,
) -> o3d.t.geometry.VoxelBlockGrid:
    """Create a tensor TSDF volume compatible with GPU execution."""
    return o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
        attr_channels=((1,), (1,), (3,)),
        voxel_size=voxel_length,
        block_resolution=BLOCK_RESOLUTION,
        block_count=BLOCK_COUNT,
        device=device,
    )


def _as_tensor_image(array: np.ndarray, device: o3d.core.Device) -> o3d.t.geometry.Image:
    """Wrap a NumPy image array as an Open3D tensor image on the target device."""
    return o3d.t.geometry.Image(o3d.core.Tensor(array, device=device))


def _as_intrinsic_tensor(
    intrinsic: np.ndarray,
    *,
    sx: float,
    sy: float,
) -> o3d.core.Tensor:
    """Scale camera intrinsics to match the RGB image resolution."""
    return o3d.core.Tensor(
        np.array(
            [
                [intrinsic[0, 0] * sx, 0.0, intrinsic[0, 2] * sx],
                [0.0, intrinsic[1, 1] * sy, intrinsic[1, 2] * sy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        dtype=o3d.core.float64,
    )


def _write_mesh_outputs(mesh: o3d.geometry.TriangleMesh, output_path: Path) -> Path:
    """Persist the fused mesh as GLB and fail loudly on export/readback issues."""
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError("TSDF fusion produced an empty mesh")

    if output_path.suffix.lower() != ".glb":
        raise ValueError(f"Expected .glb mesh output, got: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    kwargs: dict = {
        "vertices": np.asarray(mesh.vertices),
        "faces": np.asarray(mesh.triangles),
        "process": False,
    }
    if mesh.has_vertex_normals():
        kwargs["vertex_normals"] = np.asarray(mesh.vertex_normals)
    if mesh.has_vertex_colors():
        kwargs["vertex_colors"] = (
            np.asarray(mesh.vertex_colors).clip(0.0, 1.0) * 255
        ).astype(np.uint8)

    trimesh.Trimesh(**kwargs).export(output_path)

    try:
        loaded = _load_trimesh_mesh(output_path)
    except Exception as exc:
        raise ValueError(f"GLB export verification failed for {output_path}") from exc

    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError(f"GLB export verification failed for {output_path}: empty mesh")

    return output_path


def _load_trimesh_mesh(mesh_path: Path) -> trimesh.Trimesh:
    """Load a mesh path as a concrete trimesh.Trimesh for type-safe mutation."""
    return cast(trimesh.Trimesh, trimesh.load(mesh_path, force="mesh"))


def apply_sim3_to_mesh(
    mesh_path: Path,
    rot: np.ndarray,
    trans: np.ndarray,
    scale: float,
) -> Path:
    """Apply a full similarity transform to an exported mesh in place."""
    mesh = _load_trimesh_mesh(mesh_path)
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Cannot align empty mesh: {mesh_path}")
    mesh.vertices = scale * (np.asarray(mesh.vertices) @ np.asarray(rot).T) + np.asarray(trans)
    mesh.export(mesh_path)

    loaded = _load_trimesh_mesh(mesh_path)
    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError(f"Aligned mesh export verification failed for {mesh_path}: empty mesh")
    return mesh_path


def run_tsdf_fusion(
    depths: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    images_dir: Path,
    output_path: Path,
    voxel_length: float = VOXEL_LENGTH,
    sdf_trunc: float = SDF_TRUNC,
    max_depth: float = MAX_DEPTH,
) -> Path:
    """Fuse depth maps into a mesh via TSDF and export mesh assets."""
    n_frames = len(depths)
    image_paths = sorted(images_dir.glob("*.jpg"))
    if len(image_paths) < n_frames:
        raise ValueError(
            f"TSDF fusion expected {n_frames} RGB frames, found {len(image_paths)} in {images_dir}"
        )

    device = _get_tsdf_device()
    trunc_voxel_multiplier = max(sdf_trunc / voxel_length, 1.0)
    volume = _create_voxel_block_grid(
        voxel_length=voxel_length,
        device=device,
    )

    for i in range(n_frames):
        colour_bgr = cv2.imread(str(image_paths[i]))
        if colour_bgr is None:
            raise ValueError(f"TSDF fusion could not read RGB frame: {image_paths[i]}")
        colour_rgb = cv2.cvtColor(colour_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = colour_rgb.shape[:2]

        depth_hm = depths[i]
        if depth_hm.shape[:2] != (orig_h, orig_w):
            depth_hm = cv2.resize(depth_hm, (orig_w, orig_h))

        depth_uint16 = (depth_hm * 1000.0).astype(np.uint16)
        depth_img = _as_tensor_image(depth_uint16, device)
        colour_img = _as_tensor_image(colour_rgb, device)

        depth_h, depth_w = depths[i].shape[:2]
        sx = orig_w / depth_w
        sy = orig_h / depth_h
        ixt = np.asarray(intrinsics[i])
        intrinsic = _as_intrinsic_tensor(ixt, sx=sx, sy=sy)
        extr = np.asarray(extrinsics[i])
        extrinsic = o3d.core.Tensor(extr, dtype=o3d.core.float64)
        block_coords = volume.compute_unique_block_coordinates(
            depth_img,
            intrinsic,
            extrinsic,
            depth_scale=1000.0,
            depth_max=max_depth,
            trunc_voxel_multiplier=trunc_voxel_multiplier,
        )
        volume.integrate(
            block_coords,
            depth_img,
            colour_img,
            intrinsic,
            extrinsic,
            depth_scale=1000.0,
            depth_max=max_depth,
            trunc_voxel_multiplier=trunc_voxel_multiplier,
        )

    mesh = volume.extract_triangle_mesh(weight_threshold=MESH_WEIGHT_THRESHOLD)
    mesh = mesh.to_legacy()
    mesh.compute_vertex_normals()

    return _write_mesh_outputs(mesh, output_path)
