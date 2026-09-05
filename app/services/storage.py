"""Per-task artifact storage helpers."""

from pathlib import Path
from app.config import get_reconstruct_base_dir


def create_task_dirs(task_id: str) -> Path:
    root = get_reconstruct_base_dir() / task_id
    for name in ("images", "ar_exported_pose", "reconstructed_pose", "point_cloud", "mesh", "semantic", "relocalization"):
        (root / "workspace" / name).mkdir(parents=True, exist_ok=True)
    return root


def save_file(data: bytes, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(data)


def get_task_video_path(task_root: Path) -> Path:
    return task_root / "src_video.mp4"


def get_frame_images_dir(task_root: Path) -> Path:
    return task_root / "workspace" / "images"


def get_pose_output_dir(task_root: Path) -> Path:
    return task_root / "workspace" / "ar_exported_pose"


def get_mesh_dir(task_root: Path) -> Path:
    return task_root / "workspace" / "mesh"


def get_point_cloud_dir(task_root: Path) -> Path:
    return task_root / "workspace" / "point_cloud"


def get_reconstructed_pose_dir(task_root: Path) -> Path:
    return task_root / "workspace" / "reconstructed_pose"


def get_semantic_dir(task_root: Path) -> Path:
    return task_root / "workspace" / "semantic"


def get_relocalization_dir(task_root: Path) -> Path:
    return task_root / "workspace" / "relocalization"


def get_relocalization_intrinsics_path(task_root: Path) -> Path:
    return get_relocalization_dir(task_root) / "intrinsics.npz"


def get_semantic_npz_path(task_root: Path) -> Path:
    return get_semantic_dir(task_root) / "semantic_pc.npz"


def get_objects_path(task_root: Path) -> Path:
    return get_semantic_dir(task_root) / "objects.json"


def get_object_content_path(task_root: Path) -> Path:
    return get_semantic_dir(task_root) / "object_content.json"


def get_transforms_path(task_root: Path) -> Path:
    return get_relocalization_dir(task_root) / "transforms.json"
