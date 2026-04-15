"""Tests for ONNX to RKNN vision encoder conversion."""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def onnx_file(tmp_path: Path) -> str:
    """Create a dummy ONNX file and return its path."""
    onnx_path = tmp_path / "vision_encoder.onnx"
    onnx_path.write_bytes(b"dummy onnx content")
    return str(onnx_path)


@pytest.fixture()
def rknn_config(tmp_path: Path, sample_params: dict[str, Any]) -> dict[str, Any]:
    """Build a convert config with real tmp output dir."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    cfg = dict(sample_params["convert"])
    cfg["output_dir"] = str(output_dir)
    return cfg


@pytest.fixture(autouse=True)
def mock_rknn_sdk() -> MagicMock:
    """Inject a fake rknn.api module into sys.modules before each test."""
    fake_rknn_api = MagicMock()
    fake_rknn = MagicMock()
    fake_rknn_api.RKNN = fake_rknn
    with patch.dict(sys.modules, {"rknn": MagicMock(), "rknn.api": fake_rknn_api}):
        # Also clear any previously cached import of the module under test
        sys.modules.pop("pet_quantize.convert.convert_to_rknn", None)
        yield fake_rknn


def test_convert_creates_rknn_file(
    onnx_file: str, rknn_config: dict[str, Any], mock_rknn_sdk: MagicMock
) -> None:
    """Mock RKNN, verify output path ends with .rknn."""
    mock_instance = MagicMock()
    mock_instance.load_onnx.return_value = 0
    mock_instance.build.return_value = 0
    mock_instance.export_rknn.return_value = 0
    mock_rknn_sdk.return_value = mock_instance

    from pet_quantize.convert.convert_to_rknn import convert_vision_to_rknn

    result = convert_vision_to_rknn(onnx_file, rknn_config)

    assert result.endswith(".rknn")
    assert "vision" in result
    mock_instance.export_rknn.assert_called_once()


def test_convert_uses_fp16_dtype(
    onnx_file: str, rknn_config: dict[str, Any], mock_rknn_sdk: MagicMock
) -> None:
    """Verify build is called with do_quantization=False for FP16."""
    mock_instance = MagicMock()
    mock_instance.load_onnx.return_value = 0
    mock_instance.build.return_value = 0
    mock_instance.export_rknn.return_value = 0
    mock_rknn_sdk.return_value = mock_instance

    from pet_quantize.convert.convert_to_rknn import convert_vision_to_rknn

    convert_vision_to_rknn(onnx_file, rknn_config)

    _, kwargs = mock_instance.build.call_args
    assert kwargs.get("do_quantization") is False


def test_convert_missing_onnx_raises(
    tmp_path: Path, rknn_config: dict[str, Any], mock_rknn_sdk: MagicMock
) -> None:
    """Nonexistent onnx_path raises FileNotFoundError."""
    from pet_quantize.convert.convert_to_rknn import convert_vision_to_rknn

    with pytest.raises(FileNotFoundError):
        convert_vision_to_rknn(str(tmp_path / "nonexistent.onnx"), rknn_config)


def test_convert_rknn_build_failure(
    onnx_file: str, rknn_config: dict[str, Any], mock_rknn_sdk: MagicMock
) -> None:
    """RKNN build returning -1 raises RuntimeError."""
    mock_instance = MagicMock()
    mock_instance.load_onnx.return_value = 0
    mock_instance.build.return_value = -1
    mock_rknn_sdk.return_value = mock_instance

    from pet_quantize.convert.convert_to_rknn import convert_vision_to_rknn

    with pytest.raises(RuntimeError):
        convert_vision_to_rknn(onnx_file, rknn_config)
