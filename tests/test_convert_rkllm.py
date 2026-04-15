"""Tests for LLM to RKLLM W8A8 conversion."""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def rkllm_config(tmp_path: Path, sample_params: dict[str, Any]) -> dict[str, Any]:
    """Build a convert config with real tmp paths."""
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    cfg = dict(sample_params["convert"])
    cfg["weights_dir"] = str(weights_dir)
    cfg["output_dir"] = str(output_dir)
    return cfg


@pytest.fixture()
def calib_dir(tmp_path: Path) -> str:
    """Create a dummy calibration directory."""
    d = tmp_path / "calib"
    d.mkdir()
    return str(d)


@pytest.fixture(autouse=True)
def mock_rkllm_sdk() -> MagicMock:
    """Inject a fake rkllm.api module into sys.modules before each test."""
    fake_rkllm_api = MagicMock()
    fake_converter_cls = MagicMock()
    fake_rkllm_api.RKLLMConverter = fake_converter_cls
    with patch.dict(sys.modules, {"rkllm": MagicMock(), "rkllm.api": fake_rkllm_api}):
        sys.modules.pop("pet_quantize.convert.convert_to_rkllm", None)
        yield fake_converter_cls


def test_convert_creates_rkllm_file(
    rkllm_config: dict[str, Any], calib_dir: str, mock_rkllm_sdk: MagicMock
) -> None:
    """Mock RKLLMConverter, verify .rkllm path returned."""
    mock_instance = MagicMock()
    mock_rkllm_sdk.return_value = mock_instance

    from pet_quantize.convert.convert_to_rkllm import convert_llm_to_rkllm

    result = convert_llm_to_rkllm(rkllm_config, calib_dir)

    assert result.endswith(".rkllm")
    mock_instance.convert.assert_called_once()
    mock_instance.export.assert_called_once()


def test_convert_uses_w8a8_quantization(
    rkllm_config: dict[str, Any], calib_dir: str, mock_rkllm_sdk: MagicMock
) -> None:
    """Verify quantization='w8a8' is passed to RKLLMConverter."""
    mock_instance = MagicMock()
    mock_rkllm_sdk.return_value = mock_instance

    from pet_quantize.convert.convert_to_rkllm import convert_llm_to_rkllm

    convert_llm_to_rkllm(rkllm_config, calib_dir)

    _, kwargs = mock_rkllm_sdk.call_args
    assert kwargs.get("quantization") == "w8a8"


def test_convert_missing_weights_raises(
    tmp_path: Path, sample_params: dict[str, Any], calib_dir: str, mock_rkllm_sdk: MagicMock
) -> None:
    """Nonexistent weights_dir raises FileNotFoundError."""
    cfg = dict(sample_params["convert"])
    cfg["weights_dir"] = str(tmp_path / "nonexistent")
    cfg["output_dir"] = str(tmp_path / "output")

    from pet_quantize.convert.convert_to_rkllm import convert_llm_to_rkllm

    with pytest.raises(FileNotFoundError):
        convert_llm_to_rkllm(cfg, calib_dir)
