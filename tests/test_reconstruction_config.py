import pytest

from app.config import (
    get_iggt_image_size,
    get_reference_max_views,
    get_reference_retrieval_count,
    get_runtime_interval_seconds,
)


def test_iggt_default_image_size(monkeypatch):
    monkeypatch.delenv("IGGT_IMAGE_SIZE", raising=False)
    assert get_iggt_image_size() == (392, 630)


def test_reference_defaults_match_paper(monkeypatch):
    for name in ("IGGT_MAX_REFERENCE_VIEWS", "IGGT_REFERENCE_RETRIEVAL_COUNT"):
        monkeypatch.delenv(name, raising=False)
    assert get_reference_max_views() == 50
    assert get_reference_retrieval_count() == 10


def test_runtime_interval_defaults_to_ten_seconds(monkeypatch):
    monkeypatch.delenv("IGGT_RELOCALIZATION_INTERVAL_SECONDS", raising=False)
    assert get_runtime_interval_seconds() == pytest.approx(10.0)


def test_iggt_image_size_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("IGGT_IMAGE_SIZE", "0x480")
    with pytest.raises(ValueError):
        get_iggt_image_size()
