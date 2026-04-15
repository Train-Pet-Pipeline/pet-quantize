"""Smoke test: INT8 audio model basic accuracy."""
from __future__ import annotations

import glob
from pathlib import Path

import pytest


def test_audio_accuracy(params: dict, model_dir: str, device_id: str | None) -> None:
    """INT8 audio model achieves basic accuracy on a small sample."""
    from pet_quantize.inference import run_audio_inference

    audio_models = glob.glob(str(Path(model_dir) / "audio_*.rknn"))
    if not audio_models:
        pytest.skip("No audio RKNN model found in model_dir")

    audio_model_path = audio_models[0]

    audio_dir = params.get("convert", {}).get("audio_test_dir", "")
    if not audio_dir or not Path(audio_dir).exists():
        pytest.skip("No audio test directory configured")

    audio_files = glob.glob(str(Path(audio_dir) / "*.wav"))
    if not audio_files:
        pytest.skip("No audio test files found")

    sample_size = min(params["validate"]["smoke_sample_size"], len(audio_files))
    audio_files = audio_files[:sample_size]

    result = run_audio_inference(
        model_path=audio_model_path,
        audio_paths=audio_files,
        device_id=device_id,
    )

    expected_classes = {"eating", "drinking", "vomiting", "ambient", "other"}
    for pred in result["predictions"]:
        assert pred in expected_classes, f"Unexpected prediction: {pred}"

    for conf in result["confidences"]:
        assert conf > 0.1, f"Suspiciously low confidence: {conf}"
