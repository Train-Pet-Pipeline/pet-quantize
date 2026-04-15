"""Tests for vision encoder ONNX export."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def vision_config(tmp_path: Path, sample_params: dict[str, Any]) -> dict[str, Any]:
    """Build a convert config with real tmp paths."""
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    cfg = dict(sample_params["convert"])
    cfg["weights_dir"] = str(weights_dir)
    cfg["output_dir"] = str(output_dir)
    return cfg


def test_export_creates_onnx_file(vision_config: dict[str, Any]) -> None:
    """Mock AutoModel + torch.onnx.export; verify .onnx path returned and export called."""
    with (
        patch("pet_quantize.convert.export_vision_encoder.AutoModel") as mock_automodel,
        patch("pet_quantize.convert.export_vision_encoder.torch") as mock_torch,
    ):
        mock_model = MagicMock()
        mock_model.visual = MagicMock()
        mock_automodel.from_pretrained.return_value = mock_model
        mock_torch.zeros.return_value = MagicMock()

        from pet_quantize.convert.export_vision_encoder import export_vision_encoder

        result = export_vision_encoder(vision_config)

    assert result.endswith(".onnx")
    assert "vision_encoder" in result
    mock_torch.onnx.export.assert_called_once()


def test_export_uses_correct_opset(vision_config: dict[str, Any]) -> None:
    """Verify opset_version=17 is passed to torch.onnx.export."""
    with (
        patch("pet_quantize.convert.export_vision_encoder.AutoModel") as mock_automodel,
        patch("pet_quantize.convert.export_vision_encoder.torch") as mock_torch,
    ):
        mock_model = MagicMock()
        mock_model.visual = MagicMock()
        mock_automodel.from_pretrained.return_value = mock_model
        mock_torch.zeros.return_value = MagicMock()

        from pet_quantize.convert.export_vision_encoder import export_vision_encoder

        export_vision_encoder(vision_config)

    _, kwargs = mock_torch.onnx.export.call_args
    assert kwargs.get("opset_version") == 17


def test_export_missing_weights_dir(tmp_path: Path, sample_params: dict[str, Any]) -> None:
    """Nonexistent weights_dir raises FileNotFoundError."""
    cfg = dict(sample_params["convert"])
    cfg["weights_dir"] = str(tmp_path / "nonexistent")
    cfg["output_dir"] = str(tmp_path / "output")

    with patch("pet_quantize.convert.export_vision_encoder.AutoModel"):
        with patch("pet_quantize.convert.export_vision_encoder.torch"):
            from pet_quantize.convert.export_vision_encoder import export_vision_encoder

            with pytest.raises(FileNotFoundError):
                export_vision_encoder(cfg)
