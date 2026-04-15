"""Tests for pet_quantize.inference.pipeline."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def mock_rknn_rkllm_sdks() -> None:
    """Inject fake rknn/rkllm SDK modules and stub image/prompt helpers on PC."""
    fake_rknn_api = MagicMock()
    fake_rkllm_api = MagicMock()
    patches = {
        "rknn": MagicMock(),
        "rknn.api": fake_rknn_api,
        "rkllm": MagicMock(),
        "rkllm.api": fake_rkllm_api,
    }
    with patch.dict(sys.modules, patches):
        for mod in (
            "pet_quantize.inference.rknn_runner",
            "pet_quantize.inference.rkllm_runner",
            "pet_quantize.inference.pipeline",
            "pet_quantize.inference",
        ):
            sys.modules.pop(mod, None)
        with (
            patch(
                "pet_quantize.inference.pipeline._load_image",
                return_value=np.zeros((1, 3, 448, 448), dtype=np.float32),
            ),
            patch(
                "pet_quantize.inference.pipeline._build_prompt",
                return_value="mock prompt",
            ),
        ):
            yield


@pytest.fixture()
def model_dir(tmp_dir: Path) -> Path:
    """Create a dummy model directory with expected files."""
    d = tmp_dir / "converted"
    d.mkdir()
    (d / "vision_rk3576.rknn").write_bytes(b"dummy_rknn")
    (d / "qwen2vl_2b_w8a8_rk3576.rkllm").write_bytes(b"dummy_rkllm")
    return d


@pytest.fixture()
def image_paths(tmp_dir: Path) -> list[str]:
    """Create dummy image files."""
    paths = []
    for i in range(3):
        p = tmp_dir / f"image_{i}.jpg"
        p.write_bytes(b"dummy_image")
        paths.append(str(p))
    return paths


def test_pipeline_returns_expected_keys(
    model_dir: Path,
    image_paths: list[str],
    sample_params_path: Path,
) -> None:
    """Pipeline returns dict with outputs, timings, fp16_outputs keys."""
    from pet_quantize.inference.pipeline import run_quantized_pipeline

    with (
        patch("pet_quantize.inference.pipeline.RKNNRunner") as mock_rknn,
        patch("pet_quantize.inference.pipeline.RKLLMRunner") as mock_rkllm,
        patch("pet_quantize.inference.pipeline._run_fp16_reference") as mock_fp16,
    ):
        mock_rknn_inst = MagicMock()
        mock_rknn.return_value = mock_rknn_inst
        mock_rknn_inst.infer.return_value = ([np.zeros((1, 768))], 10.0)

        mock_rkllm_inst = MagicMock()
        mock_rkllm.return_value = mock_rkllm_inst
        mock_rkllm_inst.generate.return_value = ('{"valid": "json"}', 100.0)

        mock_fp16.return_value = ['{"fp16": "output"}'] * 3

        result = run_quantized_pipeline(
            model_dir=str(model_dir),
            image_paths=image_paths,
            device_id=None,
            params_path=str(sample_params_path),
        )

    assert "outputs" in result
    assert "timings" in result
    assert "fp16_outputs" in result
    assert len(result["outputs"]) == 3
    assert len(result["fp16_outputs"]) == 3


def test_pipeline_simulated_no_timings(
    model_dir: Path,
    image_paths: list[str],
    sample_params_path: Path,
) -> None:
    """In simulated mode, timings is empty list."""
    from pet_quantize.inference.pipeline import run_quantized_pipeline

    with (
        patch("pet_quantize.inference.pipeline.RKNNRunner") as mock_rknn,
        patch("pet_quantize.inference.pipeline.RKLLMRunner") as mock_rkllm,
        patch("pet_quantize.inference.pipeline._run_fp16_reference") as mock_fp16,
    ):
        mock_rknn_inst = MagicMock()
        mock_rknn.return_value = mock_rknn_inst
        mock_rknn_inst.infer.return_value = ([np.zeros((1, 768))], 10.0)

        mock_rkllm_inst = MagicMock()
        mock_rkllm.return_value = mock_rkllm_inst
        mock_rkllm_inst.generate.return_value = ('{"valid": "json"}', 100.0)

        mock_fp16.return_value = ['{"fp16": "output"}'] * 3

        result = run_quantized_pipeline(
            model_dir=str(model_dir),
            image_paths=image_paths,
            device_id=None,
            params_path=str(sample_params_path),
        )

    assert result["timings"] == []


def test_pipeline_device_mode_has_timings(
    model_dir: Path,
    image_paths: list[str],
    sample_params_path: Path,
) -> None:
    """In device mode, timings has one entry per image."""
    from pet_quantize.inference.pipeline import run_quantized_pipeline

    with (
        patch("pet_quantize.inference.pipeline.RKNNRunner") as mock_rknn,
        patch("pet_quantize.inference.pipeline.RKLLMRunner") as mock_rkllm,
        patch("pet_quantize.inference.pipeline._run_fp16_reference") as mock_fp16,
    ):
        mock_rknn_inst = MagicMock()
        mock_rknn.return_value = mock_rknn_inst
        mock_rknn_inst.infer.return_value = ([np.zeros((1, 768))], 10.0)

        mock_rkllm_inst = MagicMock()
        mock_rkllm.return_value = mock_rkllm_inst
        mock_rkllm_inst.generate.return_value = ('{"valid": "json"}', 100.0)

        mock_fp16.return_value = ['{"fp16": "output"}'] * 3

        result = run_quantized_pipeline(
            model_dir=str(model_dir),
            image_paths=image_paths,
            device_id="DEVICE123",
            params_path=str(sample_params_path),
        )

    assert len(result["timings"]) == 3


def test_pipeline_missing_model_dir(sample_params_path: Path) -> None:
    """Raises FileNotFoundError for missing model directory."""
    from pet_quantize.inference.pipeline import run_quantized_pipeline

    with pytest.raises(FileNotFoundError):
        run_quantized_pipeline(
            model_dir="/nonexistent/models",
            image_paths=["a.jpg"],
            params_path=str(sample_params_path),
        )
