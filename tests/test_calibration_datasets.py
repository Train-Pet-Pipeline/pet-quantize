"""DATASETS calibration plugin contract tests (vlm/vision/audio)."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from pet_schema.model_card import ModelCard

# Import paths exercise the gated cluster via PET_ALLOW_MISSING_SDK=1 — but these
# plugin modules don't depend on rknn/rkllm themselves, so direct import works.
from pet_quantize.plugins.datasets.audio_calibration_subset import AudioCalibrationSubset
from pet_quantize.plugins.datasets.vision_calibration_subset import VisionCalibrationSubset
from pet_quantize.plugins.datasets.vlm_calibration_subset import VlmCalibrationSubset


def _make_card(modality: str = "vision", card_id: str = "cal-test") -> ModelCard:
    """Build a minimal valid ModelCard (all ModelCard required fields, extra='forbid')."""
    return ModelCard(
        id=card_id,
        version="1.0.0",
        modality=modality,
        task="classification",
        arch="test-arch",
        training_recipe="test-recipe",
        hydra_config_sha="a" * 40,
        git_shas={"pet_quantize": "b" * 40},
        dataset_versions={"test_dataset": "v1"},
        checkpoint_uri="/tmp/ckpt",
        metrics={"accuracy": 0.5},
        gate_status="passed",
        trained_at=datetime.now(UTC),
        trained_by="test",
    )


def test_vision_writes_tensor_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "pet_quantize.calibration.vision_loader.load_calibration_images",
        lambda source_uri, num_samples: [torch.zeros(3, 224, 224) for _ in range(num_samples)],
    )
    plugin = VisionCalibrationSubset(
        source_uri="/dev/null", num_samples=4, batch_size=2, cache_dir=str(tmp_path)
    )
    out = plugin.run(_make_card(), recipe=MagicMock())
    uri = out.intermediate_artifacts["calibration_batch_uri"]
    assert Path(uri).exists()
    loaded = torch.load(uri)
    assert loaded.shape == (4, 3, 224, 224)


def test_audio_writes_tensor_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "pet_quantize.calibration.audio_loader.load_calibration_clips",
        lambda source_uri, num_samples: [torch.zeros(1, 64, 100) for _ in range(num_samples)],
    )
    plugin = AudioCalibrationSubset(
        source_uri="/dev/null", num_samples=3, batch_size=1, cache_dir=str(tmp_path)
    )
    out = plugin.run(_make_card(modality="audio"), recipe=MagicMock())
    loaded = torch.load(out.intermediate_artifacts["calibration_batch_uri"])
    assert loaded.shape == (3, 1, 64, 100)


def test_vlm_writes_tensor_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "pet_quantize.calibration.vlm_loader.load_calibration_pairs",
        lambda source_uri, num_samples: [
            torch.zeros(2048, dtype=torch.int64) for _ in range(num_samples)
        ],
    )
    plugin = VlmCalibrationSubset(
        source_uri="/dev/null", num_samples=2, batch_size=1, cache_dir=str(tmp_path)
    )
    out = plugin.run(_make_card(modality="multimodal"), recipe=MagicMock())
    loaded = torch.load(out.intermediate_artifacts["calibration_batch_uri"])
    assert loaded.shape == (2, 2048)


def test_content_addressable_cache_hits_second_call(tmp_path, monkeypatch):
    """Same (source_uri, num_samples) → same cache path; loader called once."""
    call_count = {"n": 0}

    def counting_loader(source_uri, num_samples):
        call_count["n"] += 1
        return [torch.zeros(3, 224, 224) for _ in range(num_samples)]

    monkeypatch.setattr(
        "pet_quantize.calibration.vision_loader.load_calibration_images", counting_loader
    )
    p1 = VisionCalibrationSubset(source_uri="/x", num_samples=2, cache_dir=str(tmp_path))
    p2 = VisionCalibrationSubset(source_uri="/x", num_samples=2, cache_dir=str(tmp_path))
    uri1 = p1.run(_make_card(), recipe=MagicMock()).intermediate_artifacts["calibration_batch_uri"]
    uri2 = p2.run(_make_card(), recipe=MagicMock()).intermediate_artifacts["calibration_batch_uri"]
    assert uri1 == uri2
    assert call_count["n"] == 1, "cache miss on second run — content-addressable cache broken"


def test_cache_key_varies_by_num_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "pet_quantize.calibration.vision_loader.load_calibration_images",
        lambda source_uri, num_samples: [torch.zeros(3, 224, 224) for _ in range(num_samples)],
    )
    u2 = (
        VisionCalibrationSubset(source_uri="/x", num_samples=2, cache_dir=str(tmp_path))
        .run(_make_card(), recipe=MagicMock())
        .intermediate_artifacts["calibration_batch_uri"]
    )
    u4 = (
        VisionCalibrationSubset(source_uri="/x", num_samples=4, cache_dir=str(tmp_path))
        .run(_make_card(), recipe=MagicMock())
        .intermediate_artifacts["calibration_batch_uri"]
    )
    assert u2 != u4


def test_stubs_raise_notimplemented():
    """Stubs are placeholders — calling them directly must raise, not silently return []."""
    from pet_quantize.calibration import audio_loader, vision_loader, vlm_loader

    with pytest.raises(NotImplementedError):
        vision_loader.load_calibration_images("/x", 1)
    with pytest.raises(NotImplementedError):
        vlm_loader.load_calibration_pairs("/x", 1)
    with pytest.raises(NotImplementedError):
        audio_loader.load_calibration_clips("/x", 1)
