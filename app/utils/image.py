"""Generic image helpers reused across backend services."""

from __future__ import annotations

import cv2
import numpy as np


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes into an OpenCV BGR array."""
    if not image_bytes:
        raise ValueError("Image file is empty or invalid")

    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("Image file is empty or invalid")
    return image


def compute_resize_resolution(
    orig_w: int,
    orig_h: int,
    max_pixels: int = 250_000,
    divisor: int = 14,
) -> tuple[int, int]:
    """Compute the largest aligned resolution under the pixel budget."""
    aspect_ratio = orig_w / orig_h
    max_width = int((max_pixels * aspect_ratio) ** 0.5)
    width = (max_width // divisor) * divisor

    while width >= divisor:
        height = round(width / aspect_ratio / divisor) * divisor
        if height < divisor:
            height = divisor
        if width * height <= max_pixels:
            return width, height
        width -= divisor

    return divisor, divisor
