"""Tests for audio CNN to FP16 RKNN conversion."""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Pre-inject fake rknn SDK into sys.modules so convert_audio can be imported
_fake_rknn_api = MagicMock()
_fake_rknn_cls = MagicMock()
_fake_rknn_api.RKNN = _fake_rknn_cls
sys.modules.setdefault("rknn", MagicMock())
sys.modules.setdefault("rknn.api", _fake_rknn_api)

# Now import the module under test (torch import happens once here, no re-init issues)
from pet_quantize.convert import convert_audio as _audio_mod  # noqa: E402


@pytest.fixture()
def audio_config(tmp_path: Path, sample_params: dict[str, Any]) -> dict[str, Any]:
    """Build a convert config with real tmp paths including a dummy checkpoint."""
    checkpoint = tmp_path / "audio_cnn.pt"
    checkpoint.write_bytes(b"dummy checkpoint")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    cfg = dict(sample_params["convert"])
    cfg["audio_checkpoint"] = str(checkpoint)
    cfg["output_dir"] = str(output_dir)
    return cfg


@pytest.fixture()
def mock_rknn_instance() -> MagicMock:
    """Return a configured mock RKNN instance with all return values set to 0."""
    instance = MagicMock()
    instance.load_onnx.return_value = 0
    instance.build.return_value = 0
    instance.export_rknn.return_value = 0
    return instance


def test_convert_creates_rknn_file(
    audio_config: dict[str, Any],
    mock_rknn_instance: MagicMock,
) -> None:
    """Mock torch.load + RKNN, verify output path ends with .rknn."""
    _fake_rknn_cls.return_value = mock_rknn_instance

    with patch.object(_audio_mod, "torch") as mock_torch:
        mock_torch.load.return_value = MagicMock()
        mock_torch.zeros.return_value = MagicMock()

        result = _audio_mod.convert_audio_to_rknn(audio_config)

    assert result.endswith(".rknn")
    assert "audio_cnn_fp16" in result
    mock_rknn_instance.export_rknn.assert_called_once()


def test_convert_uses_fp16_no_quantization(
    audio_config: dict[str, Any],
    mock_rknn_instance: MagicMock,
) -> None:
    """Verify build is called with do_quantization=False for FP16."""
    _fake_rknn_cls.return_value = mock_rknn_instance

    with patch.object(_audio_mod, "torch") as mock_torch:
        mock_torch.load.return_value = MagicMock()
        mock_torch.zeros.return_value = MagicMock()

        _audio_mod.convert_audio_to_rknn(audio_config)

    _, kwargs = mock_rknn_instance.build.call_args
    assert kwargs.get("do_quantization") is False
    assert "dataset" not in kwargs


def test_convert_missing_checkpoint_raises(
    tmp_path: Path,
    sample_params: dict[str, Any],
) -> None:
    """Nonexistent audio_checkpoint raises FileNotFoundError."""
    cfg = dict(sample_params["convert"])
    cfg["audio_checkpoint"] = str(tmp_path / "nonexistent.pt")
    cfg["output_dir"] = str(tmp_path / "output")

    with patch.object(_audio_mod, "torch"):
        with pytest.raises(FileNotFoundError):
            _audio_mod.convert_audio_to_rknn(cfg)
