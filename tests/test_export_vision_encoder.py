"""Tests for vision encoder ONNX export."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

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


def _make_mocks() -> tuple[MagicMock, MagicMock]:
    """Create mock torch and mock AutoModel for vision encoder tests."""
    mock_automodel = MagicMock()
    mock_model = MagicMock()
    mock_model.visual = MagicMock()
    mock_automodel.from_pretrained.return_value = mock_model

    mock_torch = MagicMock()
    mock_torch.zeros.return_value = MagicMock()
    return mock_torch, mock_automodel


def _run_with_mocks(
    mock_torch: MagicMock, mock_automodel: MagicMock, config: dict[str, Any]
) -> str:
    """Reload module with mocked torch/transformers and call export_vision_encoder."""
    fake_transformers = MagicMock(AutoModel=mock_automodel)
    saved_torch = sys.modules.get("torch")
    saved_transformers = sys.modules.get("transformers")
    try:
        sys.modules["torch"] = mock_torch
        sys.modules["transformers"] = fake_transformers
        import pet_quantize.convert.export_vision_encoder as mod

        importlib.reload(mod)
        return mod.export_vision_encoder(config)
    finally:
        # Restore original modules
        if saved_torch is not None:
            sys.modules["torch"] = saved_torch
        else:
            sys.modules.pop("torch", None)
        if saved_transformers is not None:
            sys.modules["transformers"] = saved_transformers
        else:
            sys.modules.pop("transformers", None)


def test_export_creates_onnx_file(vision_config: dict[str, Any]) -> None:
    """Mock AutoModel + torch.onnx.export; verify .onnx path returned and export called."""
    mock_torch, mock_automodel = _make_mocks()
    result = _run_with_mocks(mock_torch, mock_automodel, vision_config)

    assert result.endswith(".onnx")
    assert "vision_encoder" in result
    mock_torch.onnx.export.assert_called_once()


def test_export_uses_correct_opset(vision_config: dict[str, Any]) -> None:
    """Verify opset_version=17 is passed to torch.onnx.export."""
    mock_torch, mock_automodel = _make_mocks()
    _run_with_mocks(mock_torch, mock_automodel, vision_config)

    _, kwargs = mock_torch.onnx.export.call_args
    assert kwargs.get("opset_version") == 17


def test_export_missing_weights_dir(tmp_path: Path, sample_params: dict[str, Any]) -> None:
    """Nonexistent weights_dir raises FileNotFoundError."""
    cfg = dict(sample_params["convert"])
    cfg["weights_dir"] = str(tmp_path / "nonexistent")
    cfg["output_dir"] = str(tmp_path / "output")

    from pet_quantize.convert.export_vision_encoder import export_vision_encoder

    with pytest.raises(FileNotFoundError):
        export_vision_encoder(cfg)
