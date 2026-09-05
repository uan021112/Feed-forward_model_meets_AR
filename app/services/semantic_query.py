"""Language query over the persisted object catalog."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from app.config import get_semantic_query_min_similarity
from app.services.storage import get_object_content_path, get_objects_path


def encode_query_text(text: str, lseg_net, device: torch.device) -> np.ndarray:
    import clip
    with torch.no_grad():
        tokens = clip.tokenize([text]).to(device)
        vector = lseg_net.clip_pretrained.encode_text(tokens).float().cpu().numpy()[0]
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def load_object_catalog(task_root: Path) -> dict:
    path = get_objects_path(task_root)
    if not path.exists():
        raise FileNotFoundError(f"Object catalog not found: {path}")
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if not catalog.get("objects"):
        raise ValueError("Object catalog is empty")
    return catalog


def _load_content(task_root: Path) -> dict[str, dict]:
    path = get_object_content_path(task_root)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("objects", {})


def query_semantic_objects(task_root: Path, query_text: str, lseg_net, device: torch.device) -> dict:
    catalog = load_object_catalog(task_root)
    text_vector = encode_query_text(query_text, lseg_net, device)
    content = _load_content(task_root)
    ranked = []
    for obj in catalog["objects"]:
        feature = np.asarray(obj.get("language_feature", []), dtype=np.float32)
        if feature.shape != text_vector.shape:
            continue
        score = float(feature @ text_vector)
        result = dict(obj)
        result.pop("language_feature", None)
        result["content"] = content.get(obj["object_id"], obj.get("content", {}))
        result["match_score"] = score
        ranked.append(result)
    ranked.sort(key=lambda obj: (-obj["match_score"], obj["object_id"]))
    top_score = ranked[0]["match_score"] if ranked else 0.0
    matched = bool(ranked and top_score >= get_semantic_query_min_similarity())
    return {
        "success": True,
        "query_text": query_text,
        "matched": matched,
        "top_match_score": top_score,
        "similarity_threshold": get_semantic_query_min_similarity(),
        "coordinate_frame": "G",
        "highlight_mode": "object_anchor",
        "object": ranked[0] if matched else None,
        "candidates": ranked[:5] if matched else [],
    }


