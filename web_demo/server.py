"""Flask web demo — proxy frontend to the FastAPI reconstruction backend.

Accepts zip upload, converts to video, submits to the FastAPI backend
(running on port 6701), and serves results. No model loading.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import shutil
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path

import numpy as np
import open3d as o3d
import requests
from flask import Flask, jsonify, request, send_file, send_from_directory

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.config import get_reconstruct_base_dir  # noqa: E402

logger = logging.getLogger(__name__)

# Backend API base URL
BACKEND_URL = os.environ.get("ICXR_BACKEND_URL", "http://127.0.0.1:6701/api/v1")

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_FPS = 1.0  # frames per second for generated video


def _get_ply_path(task_id: str) -> Path:
    """Return the path to the reconstruction point cloud PLY file."""
    return get_reconstruct_base_dir() / task_id / "workspace" / "point_cloud" / "point_cloud.ply"


# ── Helpers ────────────────────────────────────────────────────────────

def _images_to_video(image_paths: list[Path], output_path: Path) -> Path:
    """Create a video file from a sorted list of image paths."""
    import cv2

    if not image_paths:
        raise ValueError("No images to convert")

    first = cv2.imread(str(image_paths[0]))
    if first is None:
        raise ValueError(f"Cannot read first image: {image_paths[0]}")
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, _FPS, (w, h))
    for ip in image_paths:
        frame = cv2.imread(str(ip))
        if frame is None:
            continue
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)
    writer.release()
    return output_path


def _proxy_request(method: str, path: str, **kwargs) -> requests.Response:
    """Forward a request to the backend API."""
    url = f"{BACKEND_URL}{path}"
    resp = requests.request(method, url, **kwargs)
    return resp


# ── App factory ────────────────────────────────────────────────────────

def create_app(args: argparse.Namespace) -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="")

    @app.route("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/3d")
    def app_3d():
        return send_from_directory(app.static_folder, "3d.html")

    @app.route("/<path:filename>")
    def static_files(filename: str):
        return send_from_directory(app.static_folder, filename)

    # ── POST /api/upload ──────────────────────────────────────────
    @app.route("/api/upload", methods=["POST"])
    def api_upload():
        if "zip_file" not in request.files:
            return jsonify({"error": "No 'zip_file' field in upload"}), 400

        zip_f = request.files["zip_file"]
        if not zip_f.filename or not zip_f.filename.lower().endswith(".zip"):
            return jsonify({"error": "File must be a .zip archive"}), 400

        image_paths: list[Path] = []
        ar_pose_path: Path | None = None
        video_path = tmp_dir / "input.mp4"

        try:
            zip_data = io.BytesIO(zip_f.read())
            try:
                with zipfile.ZipFile(zip_data, "r") as zf:
                    for name in zf.namelist():
                        basename = os.path.basename(name)
                        if not basename or basename.startswith("._") or basename.startswith("__MACOSX"):
                            continue
                        if basename.lower() == "ar_pose.csv":
                            ar_pose_path = tmp_dir / "ar_pose.csv"
                            with zf.open(name) as src, open(ar_pose_path, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                            continue
                        ext = os.path.splitext(basename)[1].lower()
                        if ext not in _IMAGE_EXTS:
                            continue
                        dest = tmp_dir / basename
                        dest.resolve().relative_to(tmp_dir.resolve())
                        with zf.open(name) as src, open(dest, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        image_paths.append(dest)
            except zipfile.BadZipFile:
                return jsonify({"error": "Invalid zip file"}), 400

            if not image_paths:
                return jsonify({"error": "Zip file contains no images"}), 400

            image_paths.sort()

            # Convert images to video
            _images_to_video(image_paths, video_path)

            # Submit to backend
            with open(video_path, "rb") as vf:
                files = {"video_file": ("input.mp4", vf, "video/mp4")}
                if ar_pose_path is not None:
                    files["ar_pose_file"] = ("ar_pose.csv", ar_pose_path.read_bytes(), "text/csv")
                data = {"video_start_timestamp": 0}
                resp = _proxy_request("POST", "/reconstruct", files=files, data=data)

            if resp.status_code != 202:
                return jsonify({"error": f"Backend rejected upload: {resp.text}"}), 502

            backend_result = resp.json()
            task_id = backend_result["data"]["task_id"]

            return jsonify({
                "task_id": task_id,
                "status": "PROCESSING",
            })

        except requests.ConnectionError:
            return jsonify({"error": "Backend unreachable — is FastAPI running on 6701?"}), 502
        except Exception as exc:
            logger.exception("Upload failed")
            return jsonify({"error": f"Upload failed: {exc}"}), 500
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── GET /api/task/<task_id>/status ────────────────────────────
    @app.route("/api/task/<task_id>/status", methods=["GET"])
    def api_task_status(task_id: str):
        try:
            resp = _proxy_request("GET", f"/task/{task_id}/status")
        except requests.ConnectionError:
            return jsonify({"error": "Backend unreachable"}), 502

        if resp.status_code != 200:
            return jsonify({"error": f"Backend error: {resp.text}"}), resp.status_code

        data = resp.json()["data"]
        # Determine if semantic point cloud is ready
        task_root = get_reconstruct_base_dir() / task_id
        result = data.get("result") or {}
        objects_path = task_root / "workspace" / "semantic" / "objects.json"
        has_semantic = result.get("scene_artifacts_ready", False) and objects_path.exists()
        return jsonify({
            "task_id": task_id,
            "status": data["status"],
            "stage": data.get("stage", ""),
            "error_message": data.get("error_message"),
            "has_semantic": has_semantic,
        })

    # ── GET /api/task/<task_id>/files/mesh.glb ────────────────────
    @app.route("/api/task/<task_id>/files/mesh.glb", methods=["GET"])
    def api_mesh_download(task_id: str):
        try:
            resp = _proxy_request("GET", f"/task/{task_id}/files/mesh.glb", stream=True)
        except requests.ConnectionError:
            return jsonify({"error": "Backend unreachable"}), 502

        if resp.status_code != 200:
            return jsonify({"error": f"Mesh not found: {resp.status_code}"}), resp.status_code

        return send_file(
            io.BytesIO(resp.content),
            mimetype="application/octet-stream",
        )

    @app.route("/api/task/<task_id>/files/<filename>", methods=["GET"])
    def api_artifact_download(task_id: str, filename: str):
        if filename not in {"semantic_mesh.glb", "objects.json", "object_content.json", "transforms.json", "intrinsics.npz"}:
            return jsonify({"error": "Artifact not found"}), 404
        try:
            resp = _proxy_request("GET", f"/task/{task_id}/files/{filename}", stream=True)
        except requests.ConnectionError:
            return jsonify({"error": "Backend unreachable"}), 502
        if resp.status_code != 200:
            return jsonify({"error": f"Artifact not found: {resp.status_code}"}), resp.status_code
        return send_file(io.BytesIO(resp.content), mimetype="application/octet-stream")

    # ── GET /api/task/<task_id>/xyz ───────────────────────────────
    @app.route("/api/task/<task_id>/xyz", methods=["GET"])
    def api_get_xyz(task_id: str):
        ply_path = _get_ply_path(task_id)
        if not ply_path.exists():
            return jsonify({"error": "Reconstruction point cloud not ready"}), 404
        pcd = o3d.io.read_point_cloud(str(ply_path))
        xyz = np.asarray(pcd.points, dtype=np.float32)
        return xyz.tobytes(), 200, {"Content-Type": "application/octet-stream"}

    # ── GET /api/task/<task_id>/colors ────────────────────────────
    @app.route("/api/task/<task_id>/colors", methods=["GET"])
    def api_get_colors(task_id: str):
        ply_path = _get_ply_path(task_id)
        if not ply_path.exists():
            return jsonify({"error": "Reconstruction point cloud not ready"}), 404
        pcd = o3d.io.read_point_cloud(str(ply_path))
        colors = np.asarray(pcd.colors, dtype=np.float32)  # (N, 3) in [0, 1]
        return colors.tobytes(), 200, {"Content-Type": "application/octet-stream"}

    # ── GET /api/task/<task_id>/meta ──────────────────────────────
    @app.route("/api/task/<task_id>/meta", methods=["GET"])
    def api_get_meta(task_id: str):
        ply_path = _get_ply_path(task_id)
        if not ply_path.exists():
            return jsonify({"error": "Reconstruction point cloud not ready"}), 404
        pcd = o3d.io.read_point_cloud(str(ply_path))
        xyz = np.asarray(pcd.points, dtype=np.float32)
        return jsonify({
            "num_points": int(xyz.shape[0]),
            "bbox_min": xyz.min(axis=0).tolist(),
            "bbox_max": xyz.max(axis=0).tolist(),
        })

    @app.route("/api/task/<task_id>/objects", methods=["GET"])
    def api_objects(task_id: str):
        try:
            resp = _proxy_request("GET", f"/task/{task_id}/objects")
        except requests.ConnectionError:
            return jsonify({"error": "Backend unreachable"}), 502
        return (resp.content, resp.status_code, {"Content-Type": "application/json"})

    @app.route("/api/task/<task_id>/objects/<object_id>/content", methods=["PUT"])
    def api_object_content(task_id: str, object_id: str):
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "Invalid JSON body"}), 400
        try:
            resp = _proxy_request("PUT", f"/task/{task_id}/objects/{object_id}/content", json=body)
        except requests.ConnectionError:
            return jsonify({"error": "Backend unreachable"}), 502
        return (resp.content, resp.status_code, {"Content-Type": "application/json"})

    @app.route("/api/task/<task_id>/objects/<object_id>", methods=["GET"])
    def api_object(task_id: str, object_id: str):
        try:
            resp = _proxy_request("GET", f"/task/{task_id}/objects/{object_id}")
        except requests.ConnectionError:
            return jsonify({"error": "Backend unreachable"}), 502
        return (resp.content, resp.status_code, {"Content-Type": "application/json"})

    @app.route("/api/task/<task_id>/relocalize", methods=["POST"])
    def api_relocalize(task_id: str):
        if "image_file" not in request.files:
            return jsonify({"error": "Missing image_file"}), 400
        files = {"image_file": (request.files["image_file"].filename or "query.jpg", request.files["image_file"].read(), request.files["image_file"].mimetype)}
        data = {key: request.form[key] for key in ("ar_pose", "intrinsics", "timestamp") if key in request.form}
        try:
            resp = _proxy_request("POST", f"/task/{task_id}/relocalize", files=files, data=data)
        except requests.ConnectionError:
            return jsonify({"error": "Backend unreachable"}), 502
        return (resp.content, resp.status_code, {"Content-Type": "application/json"})

    # ── POST /api/query ───────────────────────────────────────────
    @app.route("/api/query", methods=["POST"])
    def api_query():
        body = request.get_json(silent=True)
        if not body:
            return jsonify({"error": "Invalid JSON body"}), 400

        task_id = body.get("task_id")
        text = body.get("text", "").strip()

        if not task_id:
            return jsonify({"error": "Missing 'task_id'"}), 400
        if not text:
            return jsonify({"error": "Missing or empty 'text'"}), 400

        try:
            resp = _proxy_request(
                "POST",
                f"/task/{task_id}/voice-input",
                json={"text": text},
            )
        except requests.ConnectionError:
            return jsonify({"error": "Backend unreachable"}), 502

        if resp.status_code != 200:
            return jsonify({"error": f"Backend query failed: {resp.text}"}), resp.status_code

        result = resp.json()["data"]
        return jsonify({
            "matched": result.get("matched", False),
            "top_match_score": result.get("top_match_score", 0),
            "query_text": result.get("query_text", text),
            "object": result.get("object"),
            "candidates": result.get("candidates", []),
        })


    return app


# ── CLI & startup ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ICXR 3D Reconstruction Web Demo — Flask proxy to FastAPI backend"
    )
    parser.add_argument("--port", type=int, default=6800)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()

    # Quick connectivity check
    try:
        r = requests.get(f"{BACKEND_URL}/", timeout=3)
        print(f"[server] Backend reachable at {BACKEND_URL} ({r.status_code})")
    except requests.ConnectionError:
        print(f"[server] WARNING: Backend {BACKEND_URL} not reachable — upload will fail")
    except Exception as e:
        print(f"[server] WARNING: Backend check failed: {e}")

    app = create_app(args)
    print(f"[server] Listening on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=False)


if __name__ == "__main__":
    main()
