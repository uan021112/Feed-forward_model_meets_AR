"""Build persistent object instances and a semantic mesh from IGGT/LSeg outputs."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from app.config import (
    get_instance_cluster_eps,
    get_instance_cluster_min_samples,
    get_mesh_association_max_distance,
    get_min_instance_points,
    get_min_lseg_associations,
    get_semantic_association_max_distance,
)
from app.services.geometry import atomic_json_write
from app.services.storage import get_mesh_dir, get_objects_path, get_object_content_path, get_semantic_dir


def _union_find(n: int) -> tuple[list[int], Any]:
    parent = list(range(n))
    size = [1] * n
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a == b:
            return
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]
    return parent, (find, union)


def _cluster_instance_points(points: np.ndarray, features: np.ndarray, confidence: np.ndarray) -> list[np.ndarray]:
    if len(points) == 0:
        return []
    limit = 100_000
    original_indices = np.arange(len(points), dtype=np.int64)
    stride = max(1, int(np.ceil(len(points) / limit)))
    points = points[::stride]
    features = features[::stride]
    confidence = confidence[::stride]
    original_indices = original_indices[::stride]
    valid = np.isfinite(points).all(1) & np.isfinite(features).all(1) & np.isfinite(confidence) & (confidence > 0)
    points, features, confidence, original_indices = points[valid], features[valid], confidence[valid], original_indices[valid]
    if len(points) == 0:
        return []
    features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    tree = cKDTree(points)
    pairs = tree.query_pairs(get_instance_cluster_eps(), output_type="ndarray")
    _parent, (find, union) = _union_find(len(points))
    for a, b in pairs:
        if float(features[a] @ features[b]) >= 0.75:
            union(int(a), int(b))
    groups: dict[int, list[int]] = {}
    for index in range(len(points)):
        groups.setdefault(find(index), []).append(int(original_indices[index]))
    minimum = max(get_instance_cluster_min_samples(), get_min_instance_points())
    return [np.asarray(indices, dtype=np.int64) for indices in groups.values() if len(indices) >= minimum]


def _color_for_object(object_id: str) -> np.ndarray:
    digest = hashlib.sha256(object_id.encode()).digest()
    return np.asarray([digest[0], digest[1], digest[2], 255], dtype=np.uint8)




def _json_vec(values: np.ndarray) -> dict[str, float]:
    return {"x": float(values[0]), "y": float(values[1]), "z": float(values[2])}


def build_scene_objects(task_root: Path, world_points: np.ndarray, part_features: np.ndarray, confidence: np.ndarray, lseg_xyz: np.ndarray, lseg_features: np.ndarray) -> dict[str, Any]:
    points = np.asarray(world_points, dtype=np.float32).reshape(-1, 3)
    features = np.asarray(part_features, dtype=np.float32)
    if features.ndim == 4:
        features = features.transpose(0, 2, 3, 1).reshape(-1, features.shape[1])
    confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
    count = min(len(points), len(features), len(confidence))
    points, features, confidence = points[:count], features[:count], confidence[:count]
    clusters = _cluster_instance_points(points, features, confidence)
    if not clusters:
        raise ValueError("IGGT produced no valid object instances")
    order = sorted(range(len(clusters)), key=lambda i: (float(points[clusters[i], 0].mean()), float(points[clusters[i], 1].mean()), float(points[clusters[i], 2].mean()), -len(clusters[i])))
    clusters = [clusters[i] for i in order]

    lseg_xyz = np.asarray(lseg_xyz, dtype=np.float32).reshape(-1, 3)
    lseg_features = np.asarray(lseg_features, dtype=np.float32)
    if len(lseg_xyz) != len(lseg_features):
        raise ValueError("LSeg point and feature counts do not match")
    lseg_tree = cKDTree(lseg_xyz) if len(lseg_xyz) else None
    objects: list[dict[str, Any]] = []
    for object_index, indices in enumerate(clusters, start=1):
        object_id = f"object_{object_index:04d}"
        xyz = points[indices]
        centroid = xyz.mean(0)
        aabb_min, aabb_max = xyz.min(0), xyz.max(0)
        radius = float(np.linalg.norm(xyz - centroid, axis=1).max())
        matched_features = []
        matched_distances = []
        if lseg_tree is not None:
            distances, nearest = lseg_tree.query(xyz, distance_upper_bound=get_semantic_association_max_distance())
            valid = np.isfinite(distances) & (nearest < len(lseg_xyz))
            if valid.any():
                matched_features = lseg_features[nearest[valid]]
                matched_distances = distances[valid]
        if len(matched_features) < get_min_lseg_associations():
            continue
        language_feature = np.asarray(matched_features, dtype=np.float32).mean(0)
        language_feature /= max(float(np.linalg.norm(language_feature)), 1e-12)
        objects.append({
            "object_id": object_id,
            "point_count": int(len(indices)),
            "confidence": float(confidence[indices].mean()),
            "language_feature": language_feature.astype(float).tolist(),
            "centroid": _json_vec(centroid),
            "aabb": {"min": _json_vec(aabb_min), "max": _json_vec(aabb_max)},
            "bounding_sphere": {"center": _json_vec(centroid), "radius": radius},
            "anchor_position": _json_vec(centroid),
            "anchor_normal": {"x": 0.0, "y": 1.0, "z": 0.0},
            "face_indices": [],
            "lseg_association_count": int(len(matched_features)),
            "lseg_association_mean_distance": float(np.mean(matched_distances)),
            "content": {},
        })
    if not objects:
        raise ValueError("No object instances had enough LSeg associations")
    # Reassign IDs after dropping unassociated clusters.
    for index, obj in enumerate(objects, start=1):
        obj["object_id"] = f"object_{index:04d}"
    payload = {"format_version": 1, "coordinate_frame": "G", "objects": objects, "cluster_config": {"eps_m": get_instance_cluster_eps(), "min_samples": get_instance_cluster_min_samples(), "min_points": get_min_instance_points(), "min_lseg_associations": get_min_lseg_associations()}}
    atomic_json_write(get_objects_path(task_root), payload)
    atomic_json_write(get_object_content_path(task_root), {"format_version": 1, "objects": {}})
    return payload


def generate_semantic_mesh(task_root: Path) -> Path:
    mesh_path = get_mesh_dir(task_root) / "mesh.glb"
    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("Cannot build semantic mesh from empty mesh")
    data = json.loads(get_objects_path(task_root).read_text())
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_centers = vertices[faces].mean(axis=1)
    object_count = len(data["objects"])
    scores = np.full((object_count, len(faces)), -np.inf, dtype=np.float32)
    max_distance = get_mesh_association_max_distance()
    for object_index, obj in enumerate(data["objects"]):
        center = np.array([obj["centroid"][axis] for axis in ("x", "y", "z")], dtype=np.float32)
        radius = float(obj.get("bounding_sphere", {}).get("radius", 0.0))
        distances = np.linalg.norm(face_centers - center, axis=1)
        valid = distances <= radius + max_distance
        scores[object_index, valid] = float(obj.get("confidence", 1.0)) * np.maximum(0.0, 1.0 - distances[valid] / max(radius + max_distance, 1e-6))
    face_labels = np.full(len(faces), -1, dtype=np.int32)
    if object_count:
        order = np.argsort(scores, axis=0)
        best = order[-1]
        best_scores = scores[best, np.arange(len(faces))]
        second_scores = scores[order[-2], np.arange(len(faces))] if object_count > 1 else np.zeros(len(faces), dtype=np.float32)
        confident = np.isfinite(best_scores) & (best_scores > 0) & (best_scores >= 0.6 * np.maximum(best_scores + np.maximum(second_scores, 0), 1e-12))
        face_labels[confident] = best[confident]
    colors = np.full((len(vertices), 4), [128, 128, 128, 255], dtype=np.uint8)
    for object_index, obj in enumerate(data["objects"]):
        chosen = np.flatnonzero(face_labels == object_index)
        if len(chosen) == 0:
            raise ValueError(f"Object {obj['object_id']} has no associated mesh faces")
        obj["face_indices"] = chosen.astype(int).tolist()
        chosen_centers = face_centers[chosen]
        center = np.array([obj["centroid"][axis] for axis in ("x", "y", "z")], dtype=np.float32)
        anchor_index = int(np.argmin(np.linalg.norm(chosen_centers - center, axis=1)))
        obj["anchor_position"] = _json_vec(chosen_centers[anchor_index])
        normals = np.cross(vertices[faces[chosen, 1]] - vertices[faces[chosen, 0]], vertices[faces[chosen, 2]] - vertices[faces[chosen, 0]])
        normal = normals.mean(axis=0)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        obj["anchor_normal"] = _json_vec(normal)
        colors[faces[chosen]] = _color_for_object(obj["object_id"])
    semantic = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)
    output = get_mesh_dir(task_root) / "semantic_mesh.glb"
    semantic.export(output)
    atomic_json_write(get_objects_path(task_root), data)
    return output
