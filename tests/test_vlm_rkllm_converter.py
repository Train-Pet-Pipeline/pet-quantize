"""VlmRkllmW4A16Converter plugin tests."""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from pet_schema.model_card import ModelCard


def _make_vlm_card(cal_uri: str | None = "/tmp/cal.pt") -> ModelCard:
    """Build a minimal valid ModelCard with optional calibration batch URI."""
    intermediate = {"calibration_batch_uri": cal_uri} if cal_uri else {}
    return ModelCard(
        id="vlm-test",
        version="1.0.0",
        modality="vision",
        task="vlm",
        arch="qwen2vl-2b",
        training_recipe="test-recipe",
        hydra_config_sha="a" * 40,
        git_shas={"pet_quantize": "b" * 40},
        dataset_versions={"test": "v1"},
        checkpoint_uri="/tmp/vlm-ckpt",
        intermediate_artifacts=intermediate,
        metrics={"accuracy": 0.9},
        gate_status="passed",
        trained_at=datetime.now(UTC),
        trained_by="test",
    )


def test_converter_registered() -> None:
    """Plugin registers under CONVERTERS['vlm_rkllm_w4a16'] after module import."""
    from pet_infra.registry import CONVERTERS

    import pet_quantize.plugins.converters.vlm_rkllm_w4a16  # noqa: F401

    assert "vlm_rkllm_w4a16" in CONVERTERS.module_dict


def test_converter_calls_sdk_and_emits_edge_artifact(tmp_path, monkeypatch) -> None:
    """Happy path: SDK wrapper is called, EdgeArtifact + QuantConfig set on card."""
    mock_output = tmp_path / "qwen2vl.rkllm"
    mock_output.write_bytes(b"fake rkllm contents")

    def fake_convert(config: dict, calib_dir: str) -> str:
        assert config["weights_dir"] == "/tmp/vlm-ckpt"
        assert config["llm"]["rkllm_target"] == "rk3576"
        assert config["llm"]["quantization"] == "w4a16"
        assert calib_dir == "/tmp/cal.pt"
        return str(mock_output)

    import pet_quantize.convert.convert_to_rkllm as _mod  # ensure in sys.modules before patch
    monkeypatch.setattr(_mod, "convert_llm_to_rkllm", fake_convert)

    from pet_quantize.plugins.converters.vlm_rkllm_w4a16 import VlmRkllmW4A16Converter

    plugin = VlmRkllmW4A16Converter(
        target_platform="rk3576",
        quantized_dtype="w4a16",
        output_dir=str(tmp_path),
    )
    out = plugin.run(_make_vlm_card(), recipe=MagicMock())

    assert len(out.edge_artifacts) == 1
    artifact = out.edge_artifacts[0]
    assert artifact.format == "rkllm"
    assert artifact.target_hardware == ["rk3576"]
    assert artifact.artifact_uri.endswith(".rkllm")
    assert len(artifact.sha256) == 64

    assert out.quantization is not None
    assert out.quantization.method == "ptq_int8"
    assert out.quantization.bits == 4
    assert out.quantization.calibration_dataset_uri == "/tmp/cal.pt"


def test_converter_requires_calibration_batch_in_card(tmp_path) -> None:
    """Raise ValueError when intermediate_artifacts lacks calibration_batch_uri."""
    from pet_quantize.plugins.converters.vlm_rkllm_w4a16 import VlmRkllmW4A16Converter

    plugin = VlmRkllmW4A16Converter(
        target_platform="rk3576",
        quantized_dtype="w4a16",
        output_dir=str(tmp_path),
    )
    with pytest.raises(ValueError, match="calibration_batch_uri"):
        plugin.run(_make_vlm_card(cal_uri=None), recipe=MagicMock())
