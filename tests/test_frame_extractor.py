"""Regression tests for frame extraction helpers."""

import numpy as np

from app.services.frame_extractor import _is_black_frame, compute_resize_resolution


def test_compute_resize_resolution_matches_current_1080p_budgeted_output():
    assert compute_resize_resolution(1920, 1080) == (658, 364)


def test_compute_resize_resolution_keeps_divisibility_and_pixel_budget():
    width, height = compute_resize_resolution(4032, 3024)

    assert (width, height) == (574, 434)
    assert width % 14 == 0
    assert height % 14 == 0
    assert width * height <= 250_000


class TestBlackFrameDetection:
    def test_pure_black_frame_is_black(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert _is_black_frame(frame) is True

    def test_very_dark_but_not_black_is_black(self):
        frame = np.full((480, 640, 3), 9, dtype=np.uint8)
        assert _is_black_frame(frame) is True

    def test_frame_at_threshold_is_not_black(self):
        frame = np.full((480, 640, 3), 11, dtype=np.uint8)
        assert _is_black_frame(frame) is False

    def test_bright_frame_is_not_black(self):
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        assert _is_black_frame(frame) is False
    def test_mixed_frame_with_dark_region_is_not_black(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[160:320, 200:400] = 255  # 200x160 white patch (~10% of frame)
        assert _is_black_frame(frame) is False
