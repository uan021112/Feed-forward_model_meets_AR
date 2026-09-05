"""IGGT deployment, object-service, and runtime pose APIs."""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import get_deployment_alignment_max_rms, get_reconstruct_base_dir, get_runtime_alignment_max_rms
from app.services.geometry import (
    atomic_json_write,
    load_ar_poses,
    save_reconstruction_intrinsics,
    save_reconstruction_point_cloud,
    save_reconstruction_pose_files,
    transform_c2w_by_sim3,
    transform_w2c_by_sim3,
    weighted_sim3_alignment,
)
from app.services.storage import (
    create_task_dirs,
    get_frame_images_dir,
    get_mesh_dir,
    get_object_content_path,
    get_objects_path,
    get_point_cloud_dir,
    get_pose_output_dir,
    get_reconstructed_pose_dir,
    get_relocalization_intrinsics_path,
    get_relocalization_dir,
    get_semantic_dir,
    get_task_video_path,
    get_transforms_path,
    save_file,
)
from app.services.task_store import create_task, get_task, update_status
from app.utils.validation import validate_ar_pose_csv

router = APIRouter()
logger = logging.getLogger(__name__)
_ALLOWED_DOWNLOADS = {"mesh.glb", "semantic_mesh.glb", "objects.json", "object_content.json", "transforms.json", "intrinsics.npz", "point_cloud.ply"}


def _confidence_weights(confidence: np.ndarray) -> np.ndarray:
    values = np.asarray(confidence, dtype=np.float64).reshape(len(confidence), -1)
    weights = np.nanmean(np.where(np.isfinite(values) & (values > 0), values, np.nan), axis=1)
    return np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)


def _apply_sim3_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray, scale: float) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    return (float(scale) * (values @ rotation.T) + np.asarray(translation, dtype=np.float32)).astype(np.float32)


def _identity_transform(metric: bool) -> dict[str, Any]:
    return {"format_version": 1, "coordinate_frame": "G", "metric_scale": metric, "coordinate_scale": "metric" if metric else "model", "map_to_ar": None if not metric else {"matrix": np.eye(4).tolist(), "position": {"x": 0.0, "y": 0.0, "z": 0.0}, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}}

def _load_camera_intrinsics(task_root: Path, image_paths: list[Path], predicted: np.ndarray, model_image_size: tuple[int, int]) -> tuple[np.ndarray, str]:
    path = task_root / "camera_intrinsics.json"
    if not path.exists():
        return np.asarray(predicted, dtype=np.float32), "iggt_predicted"
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("intrinsics", payload.get("matrix", payload)) if isinstance(payload, dict) else payload
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.shape == (3, 3):
        matrix = np.repeat(matrix[None, ...], len(image_paths), axis=0)
    if matrix.shape != (len(image_paths), 3, 3) or not np.isfinite(matrix).all() or np.any(matrix[:, 0, 0] <= 0) or np.any(matrix[:, 1, 1] <= 0):
        raise ValueError("camera_intrinsics_file must contain one valid 3x3 matrix per image")
    model_width, model_height = model_image_size
    for index, path in enumerate(image_paths):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        sx, sy = model_width / float(image.shape[1]), model_height / float(image.shape[0])
        matrix[index, 0, 0] *= sx
        matrix[index, 0, 2] *= sx
        matrix[index, 1, 1] *= sy
        matrix[index, 1, 2] *= sy
    return matrix, "ar_device"

def _lseg_device() -> torch.device:
    configured = os.environ.get("LSEG_DEVICE", "cuda:0")
    if configured.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(configured)


async def _process_task_background(task_id: str, video_start_timestamp: int) -> None:
    try:
        update_status(task_id, status="PROCESSING", stage="extracting frames")
        task_root = create_task_dirs(task_id)
        video_path = get_task_video_path(task_root)
        images_dir = get_frame_images_dir(task_root)
        ar_pose_path = task_root / "ar_pose.csv"
        pose_dir = get_pose_output_dir(task_root)
        frame_extractor = __import__("app.services.frame_extractor", fromlist=["extract_frames", "extract_frames_at_timestamps"])
        if ar_pose_path.exists():
            import csv
            with ar_pose_path.open() as fh:
                timestamps = [int(row["timestamp"]) for row in csv.DictReader(fh)]
            frame_results = frame_extractor.extract_frames_at_timestamps(video_path, images_dir, video_start_timestamp, timestamps)
        else:
            frame_results = frame_extractor.extract_frames(video_path, images_dir, video_start_timestamp)
        frame_timestamps = [ts for ts, _ in frame_results]
        if ar_pose_path.exists():
            shutil.copy2(ar_pose_path, pose_dir / "ar_pose.csv")
        ar_extrinsics = load_ar_poses(pose_dir, frame_timestamps) if ar_pose_path.exists() else None

        update_status(task_id, status="PROCESSING", stage="IGGT geometry, camera, confidence, and instance inference")
        from app.services.iggt import run_iggt
        result = run_iggt(images_dir)
        image_paths = sorted(images_dir.glob("*.jpg"))
        depths, extrinsics, intrinsics, confidence = result.depths, result.extrinsics, result.intrinsics, result.confidence
        intrinsics, intrinsics_source = _load_camera_intrinsics(task_root, image_paths, intrinsics, (int(depths.shape[2]), int(depths.shape[1])))
        weights = _confidence_weights(confidence)
        sim3 = None
        metric = ar_extrinsics is not None
        if ar_extrinsics is not None:
            model_c2w = np.linalg.inv(extrinsics)
            ar_c2w = np.linalg.inv(ar_extrinsics)
            rotation, translation, scale, rms, residuals = weighted_sim3_alignment(model_c2w[:, :3, 3], ar_c2w[:, :3, 3], weights)
            if rms > get_deployment_alignment_max_rms():
                raise ValueError(f"Deployment AR alignment residual {rms:.6f} m exceeds {get_deployment_alignment_max_rms():.6f} m")
            sim3 = (rotation, translation, scale, rms, residuals)
            extrinsics = transform_w2c_by_sim3(extrinsics, rotation, translation, scale)
            transformed_world = _apply_sim3_points(result.world_points, rotation, translation, scale)
        else:
            transformed_world = result.world_points
        save_reconstruction_intrinsics(image_paths, get_relocalization_intrinsics_path(task_root), intrinsics, intrinsics_source, model_image_size=(int(depths.shape[2]), int(depths.shape[1])))
        np.savez_compressed(get_relocalization_dir(task_root) / "view_confidence.npz", frame_ids=np.asarray([path.stem for path in image_paths]), reliability=_confidence_weights(confidence).astype(np.float32))

        update_status(task_id, status="PROCESSING", stage="LSeg semantic object extraction")
        from app.services.semantic import extract_semantic_point_cloud, get_lseg_model, cache_semantic_point_cloud
        lseg = get_lseg_model()
        if lseg is None:
            raise ValueError("LSeg model is required for AR-ready object semantics")
        semantic_xyz, semantic_features = extract_semantic_point_cloud(sorted(images_dir.glob("*.jpg")), depths, intrinsics, extrinsics, confidence, lseg, _lseg_device())
        if len(semantic_xyz) == 0:
            raise ValueError("LSeg produced no valid semantic points")
        semantic_dir = get_semantic_dir(task_root)
        semantic_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(semantic_dir / "semantic_pc.npz", xyz=semantic_xyz, features=semantic_features)
        cache_semantic_point_cloud(task_id, semantic_xyz, semantic_features)

        update_status(task_id, status="PROCESSING", stage="metric point cloud and mesh fusion")
        save_reconstruction_pose_files(frame_timestamps, extrinsics, get_reconstructed_pose_dir(task_root))
        save_reconstruction_point_cloud(images_dir, depths, intrinsics, extrinsics, get_point_cloud_dir(task_root) / "point_cloud.ply", confidence=confidence)
        from app.services.tsdf_fusion import run_tsdf_fusion
        mesh_path = get_mesh_dir(task_root) / "mesh.glb"
        run_tsdf_fusion(depths, extrinsics, intrinsics, images_dir, mesh_path)

        from app.services.scene_semantics import build_scene_objects, generate_semantic_mesh
        build_scene_objects(task_root, transformed_world, result.part_features, confidence, semantic_xyz, semantic_features)
        generate_semantic_mesh(task_root)
        from app.services.relocalization import initialize_reference_view_pool
        reference_view_count = initialize_reference_view_pool(task_root)
        transforms = _identity_transform(metric)
        if sim3 is not None:
            transforms.update({"deployment_alignment": {"scale": sim3[2], "weighted_rms_m": sim3[3], "max_residual_m": float(np.max(sim3[4])), "paired_frame_count": int(len(sim3[4]))}})
        atomic_json_write(get_transforms_path(task_root), transforms)
        result_payload = {"num_frames": len(frame_results), "mesh_path": str(mesh_path), "has_ar_pose": metric, "metric_scale": metric, "coordinate_scale": "metric" if metric else "model", "scene_artifacts_ready": True, "semantic_artifacts_ready": True, "reference_view_count": reference_view_count, "deployment_alignment": transforms.get("deployment_alignment")}
        update_status(task_id, status="SUCCESS", stage="Completed", result=result_payload)
    except ValueError as exc:
        update_status(task_id, status="ERROR", error_message=str(exc))
    except Exception as exc:
        logger.exception("Reconstruction failed for task %s", task_id)
        update_status(task_id, status="ERROR", error_message=f"Unexpected error during processing: {exc}")


@router.post("/reconstruct", status_code=202)
async def submit_reconstruct_task(background_tasks: BackgroundTasks, video_file: UploadFile = File(...), ar_pose_file: UploadFile | None = File(None), camera_intrinsics_file: UploadFile | None = File(None), video_start_timestamp: int = Form(...)):
    task_id = f"rec_{uuid.uuid4().hex[:10]}"
    task_root = create_task_dirs(task_id)
    video_bytes = await video_file.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Video file is empty")
    save_file(video_bytes, get_task_video_path(task_root))
    ar_pose_bytes = None
    if ar_pose_file is not None:
        ar_pose_bytes = await ar_pose_file.read()
        try:
            validate_ar_pose_csv(ar_pose_bytes)
        except ValueError as exc:
            shutil.rmtree(task_root, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        save_file(ar_pose_bytes, task_root / "ar_pose.csv")
    intrinsics_bytes = None
    if camera_intrinsics_file is not None:
        intrinsics_bytes = await camera_intrinsics_file.read()
        try:
            json.loads(intrinsics_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            shutil.rmtree(task_root, ignore_errors=True)
            raise HTTPException(status_code=400, detail="camera_intrinsics_file must be valid JSON") from exc
        save_file(intrinsics_bytes, task_root / "camera_intrinsics.json")
    task = create_task(task_id=task_id, estimated_time_cost=300, video_start_timestamp=video_start_timestamp)
    background_tasks.add_task(_process_task_background, task_id, video_start_timestamp)
    return {"code": 200, "message": "Task submitted successfully.", "data": {"task_id": task_id, "status": task["status"], "estimated_time_cost": task["estimated_time_cost"], "created_at": task["created_at"], "video_start_timestamp": video_start_timestamp, "has_ar_pose": ar_pose_file is not None, "has_camera_intrinsics": camera_intrinsics_file is not None}}


@router.get("/task/{task_id}/status")
async def get_task_status(task_id: str):
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"code": 200, "message": "Success", "data": task}


@router.get("/task/{task_id}/files/{filename}")
async def download_task_file(task_id: str, filename: str):
    if filename not in _ALLOWED_DOWNLOADS:
        raise HTTPException(status_code=404, detail="File not found")
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    task_root = get_reconstruct_base_dir() / task_id
    paths = {"mesh.glb": get_mesh_dir(task_root) / "mesh.glb", "semantic_mesh.glb": get_mesh_dir(task_root) / "semantic_mesh.glb", "objects.json": get_objects_path(task_root), "object_content.json": get_object_content_path(task_root), "transforms.json": get_transforms_path(task_root), "intrinsics.npz": get_relocalization_intrinsics_path(task_root), "point_cloud.ply": get_point_cloud_dir(task_root) / "point_cloud.ply"}
    path = paths[filename]
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


class RuntimePosition(BaseModel):
    x: float
    y: float
    z: float


class RuntimeRotation(BaseModel):
    x: float
    y: float
    z: float
    w: float


class RuntimePose(BaseModel):
    position: RuntimePosition
    rotation: RuntimeRotation

class ObjectContent(BaseModel):
    annotation: dict[str, Any] | None = None
    virtual_content: dict[str, Any] | None = None
    explanation: dict[str, Any] | None = None

def _validate_content(content: dict[str, Any]) -> None:
    virtual = content.get("virtual_content")
    if not isinstance(virtual, dict):
        return
    asset_url = virtual.get("asset_url")
    if asset_url is None:
        return
    if not isinstance(asset_url, str) or asset_url.startswith(("file:", "/")) or ".." in asset_url or ("://" in asset_url and not asset_url.startswith("https://")):
        raise HTTPException(status_code=400, detail="asset_url must be a task-relative path or HTTPS URL")
    def ensure_finite(value: Any) -> None:
        if isinstance(value, (int, float)) and not np.isfinite(value):
            raise HTTPException(status_code=400, detail="content contains a non-finite number")
        if isinstance(value, dict):
            for child in value.values():
                ensure_finite(child)
        if isinstance(value, list):
            for child in value:
                ensure_finite(child)
    ensure_finite(content)


def _runtime_c2w(pose: RuntimePose) -> np.ndarray:
    p, q = pose.position, pose.rotation
    values = np.array([q.w, q.x, q.y, q.z], dtype=np.float64)
    norm = np.linalg.norm(values)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("AR pose quaternion is invalid")
    w, x, y, z = values / norm
    rotation = np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)], [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)], [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]])
    c2w = np.eye(4)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = [p.x, p.y, p.z]
    if not np.isfinite(c2w).all():
        raise ValueError("AR pose contains non-finite values")
    return c2w


@router.post("/task/{task_id}/relocalize")
async def relocalize_task_image(task_id: str, image_file: UploadFile = File(...), ar_pose: str | None = Form(None), intrinsics: str | None = Form(None), timestamp: int | None = Form(None)):
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] != "SUCCESS":
        raise HTTPException(status_code=400, detail=f"Task scene is not ready for relocalization. Status: {task['status']}")
    from app.services.relocalization import decode_query_image, relocalize_query_image
    try:
        query_image = decode_query_image(await image_file.read())
        runtime_pose = RuntimePose.model_validate(json.loads(ar_pose)) if ar_pose else None
        query_intrinsics = np.asarray(json.loads(intrinsics), dtype=np.float32) if intrinsics else None
        result = relocalize_query_image(get_reconstruct_base_dir() / task_id, query_image, runtime_c2w=(_runtime_c2w(runtime_pose) if runtime_pose else None), query_intrinsics=query_intrinsics, timestamp=timestamp)
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"code": 200, "message": "Pose refinement successful." if result.get("success") else "Pose refinement rejected.", "data": result}


@router.post("/task/{task_id}/voice-input")
async def voice_input_query(task_id: str, body: dict[str, str]):
    query_text = body.get("text", "").strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Text is empty")
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] != "SUCCESS":
        raise HTTPException(status_code=409, detail="Task is not completed yet")
    from app.services.semantic import get_lseg_model
    lseg = get_lseg_model()
    if lseg is None:
        raise HTTPException(status_code=503, detail="LSeg model not available")
    from app.services.semantic_query import query_semantic_objects
    try:
        data = query_semantic_objects(get_reconstruct_base_dir() / task_id, query_text, lseg, _lseg_device())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"code": 200, "message": "Success", "data": data}


@router.get("/task/{task_id}/objects")
async def list_objects(task_id: str):
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] != "SUCCESS":
        raise HTTPException(status_code=409, detail="Task is not completed yet")
    task_root = get_reconstruct_base_dir() / task_id
    if not get_objects_path(task_root).exists():
        raise HTTPException(status_code=404, detail="Object catalog not found")
    return json.loads(get_objects_path(task_root).read_text())


@router.get("/task/{task_id}/objects/{object_id}")
async def get_object(task_id: str, object_id: str):
    catalog = await list_objects(task_id)
    for obj in catalog["objects"]:
        if obj["object_id"] == object_id:
            content_path = get_object_content_path(get_reconstruct_base_dir() / task_id)
            content = json.loads(content_path.read_text()).get("objects", {}).get(object_id, {}) if content_path.exists() else {}
            obj["content"] = content
            return obj
    raise HTTPException(status_code=404, detail="Object not found")


@router.put("/task/{task_id}/objects/{object_id}/content")
async def put_object_content(task_id: str, object_id: str, body: ObjectContent):
    content = body.model_dump(exclude_none=True)
    if not content:
        raise HTTPException(status_code=400, detail="At least one content field is required")
    task_root = get_reconstruct_base_dir() / task_id
    catalog = await list_objects(task_id)
    if object_id not in {obj["object_id"] for obj in catalog["objects"]}:
        raise HTTPException(status_code=404, detail="Object not found")
    path = get_object_content_path(task_root)
    payload = json.loads(path.read_text()) if path.exists() else {"format_version": 1, "objects": {}}
    _validate_content(content)
    payload.setdefault("objects", {})[object_id] = content
    atomic_json_write(path, payload)
    return await get_object(task_id, object_id)
