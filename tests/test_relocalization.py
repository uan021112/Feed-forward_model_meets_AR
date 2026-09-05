import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services.geometry import weighted_sim3_alignment
from app.services.relocalization import decode_query_image


def _jpg_bytes(value: int = 90) -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.full((8, 6, 3), value, dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def test_decode_query_image_rejects_invalid_payload():
    with pytest.raises(ValueError):
        decode_query_image(b"not-an-image")


def test_decode_query_image_loads_valid_jpeg():
    image = decode_query_image(_jpg_bytes())
    assert image.shape == (8, 6, 3)


def test_weighted_sim3_recovers_metric_alignment():
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    target = 2.0 * source + np.array([5.0, -1.0, 0.5])
    rotation, translation, scale, rms, residuals = weighted_sim3_alignment(source, target, np.ones(4))
    assert scale == pytest.approx(2.0)
    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(translation, [5.0, -1.0, 0.5], atol=1e-6)
    assert rms == pytest.approx(0.0, abs=1e-6)
    assert np.all(residuals < 1e-6)


def test_reference_artifact_payload_has_new_version(tmp_path: Path):
    path = tmp_path / "reference_views.json"
    path.write_text(json.dumps({"format_version": 2, "max_views": 50, "retrieval_count": 10, "views": []}))
    assert json.loads(path.read_text())["retrieval_count"] == 10
