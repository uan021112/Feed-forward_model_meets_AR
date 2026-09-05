"""Regression tests for TSDF mesh export."""

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pytest
import trimesh

from app.services import tsdf_fusion
from app.services.tsdf_fusion import apply_sim3_to_mesh, _write_mesh_outputs, run_tsdf_fusion


def test_write_mesh_outputs_writes_readable_glb_only(tmp_path: Path):
    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.compute_vertex_normals()

    output_path = tmp_path / "mesh.glb"
    result = _write_mesh_outputs(mesh, output_path)

    assert result == output_path
    assert output_path.exists()
    assert not (tmp_path / "mesh.ply").exists()

    loaded = trimesh.load(output_path, force="mesh")
    assert len(loaded.vertices) > 0
    assert len(loaded.faces) > 0


def test_write_mesh_outputs_rejects_empty_mesh(tmp_path: Path):
    mesh = o3d.geometry.TriangleMesh()

    with pytest.raises(ValueError, match="empty mesh"):
        _write_mesh_outputs(mesh, tmp_path / "mesh.glb")


def test_write_mesh_outputs_raises_when_glb_verification_fails(tmp_path: Path, monkeypatch):
    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.compute_vertex_normals()

    def broken_load(*args, **kwargs):
        raise RuntimeError("bad glb")

    monkeypatch.setattr(trimesh, "load", broken_load)

    with pytest.raises(ValueError, match="GLB export verification failed"):
        _write_mesh_outputs(mesh, tmp_path / "mesh.glb")


def test_apply_sim3_to_mesh_applies_full_transform_to_vertices(tmp_path: Path):
    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.compute_vertex_normals()
    output_path = _write_mesh_outputs(mesh, tmp_path / "mesh.glb")

    rot = np.eye(3, dtype=np.float32)
    trans = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    scale = 2.0

    apply_sim3_to_mesh(output_path, rot, trans, scale)

    loaded = trimesh.load(output_path, force="mesh")
    mins = loaded.vertices.min(axis=0)
    maxs = loaded.vertices.max(axis=0)
    np.testing.assert_allclose(mins, trans, atol=1e-5)
    np.testing.assert_allclose(maxs, trans + scale, atol=1e-5)


def test_run_tsdf_fusion_uses_tensor_voxel_block_grid_and_writes_mesh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for index in range(3):
        image = np.full((16, 16, 3), 255 - index * 20, dtype=np.uint8)
        assert cv2.imwrite(str(images_dir / f"{index:03d}.jpg"), image)

    depths = np.stack(
        [
            np.full((16, 16), 1.0, dtype=np.float32),
            np.full((16, 16), 1.1, dtype=np.float32),
            np.full((16, 16), 0.9, dtype=np.float32),
        ]
    )
    extrinsics = np.stack([np.eye(4, dtype=np.float64) for _ in range(3)])
    intrinsics = np.stack(
        [
            np.array(
                [[20.0, 0.0, 8.0], [0.0, 20.0, 8.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            for _ in range(3)
        ]
    )

    def fail_if_legacy_tsdf_used(*args, **kwargs):
        raise AssertionError("legacy ScalableTSDFVolume should not be used")

    monkeypatch.setattr(
        tsdf_fusion.o3d.pipelines.integration,
        "ScalableTSDFVolume",
        fail_if_legacy_tsdf_used,
    )

    output_path = tmp_path / "mesh.glb"
    result = run_tsdf_fusion(
        depths,
        extrinsics,
        intrinsics,
        images_dir,
        output_path,
        voxel_length=0.05,
        sdf_trunc=0.4,
        max_depth=3.0,
    )

    assert result == output_path
    assert output_path.exists()

    loaded = trimesh.load(output_path, force="mesh")
    assert len(loaded.vertices) > 0
    assert len(loaded.faces) > 0


def test_run_tsdf_fusion_rejects_unreadable_rgb_frame(tmp_path: Path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "000.jpg").write_bytes(b"not-a-real-jpeg")

    depths = np.full((1, 16, 16), 1.0, dtype=np.float32)
    extrinsics = np.expand_dims(np.eye(4, dtype=np.float64), axis=0)
    intrinsics = np.expand_dims(
        np.array([[20.0, 0.0, 8.0], [0.0, 20.0, 8.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        axis=0,
    )

    with pytest.raises(ValueError, match="could not read RGB frame"):
        run_tsdf_fusion(depths, extrinsics, intrinsics, images_dir, tmp_path / "mesh.glb")
