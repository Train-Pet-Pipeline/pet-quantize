"""VisionRknnFp16Converter plugin tests."""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from pet_schema.model_card import ModelCard


def _make_vision_card() -> ModelCard:
    """Build a minimal valid ModelCard for vision conversion."""
    return ModelCard(
        id="vision-test",
        version="1.0.0",
        modality="vision",
        task="classification",
        arch="qwen2vl-vision-encoder",
        training_recipe="test-recipe",
        hydra_config_sha="a" * 40,
        git_shas={"pet_quantize": "b" * 40},
        dataset_versions={"test": "v1"},
        checkpoint_uri="/tmp/vision-ckpt",
        metrics={"accuracy": 0.9},
        gate_status="passed",
        trained_at=datetime.now(UTC),
        trained_by="test",
    )


def test_converter_registered() -> None:
    """Plugin registers under CONVERTERS['vision_rknn_fp16'] after module import."""
    from pet_infra.registry import CONVERTERS

    import pet_quantize.plugins.converters.vision_rknn_fp16  # noqa: F401

    assert "vision_rknn_fp16" in CONVERTERS.module_dict


def test_converter_chains_export_and_convert(tmp_path, monkeypatch) -> None:
    """Happy path: exports ONNX then converts to RKNN; both wrappers invoked."""
    onnx_out = tmp_path / "vision.onnx"
    onnx_out.write_bytes(b"fake onnx")
    rknn_out = tmp_path / "vision.rknn"
    rknn_out.write_bytes(b"fake rknn")

    from pet_quantize.convert import convert_to_rknn as _rknn_mod
    from pet_quantize.convert import export_vision_encoder as _export_mod

    calls = {"export": 0, "convert": 0, "onnx_received": None}

    def fake_export(config: dict) -> str:
        calls["export"] += 1
        assert config["weights_dir"] == "/tmp/vision-ckpt"
        assert config["vision"]["rknn_target"] == "rk3576"
        return str(onnx_out)

    def fake_convert(onnx_path: str, config: dict, calib_dir=None) -> str:
        calls["convert"] += 1
        calls["onnx_received"] = onnx_path
        assert config["vision"]["rknn_target"] == "rk3576"
        return str(rknn_out)

    monkeypatch.setattr(_export_mod, "export_vision_encoder", fake_export)
    monkeypatch.setattr(_rknn_mod, "convert_vision_to_rknn", fake_convert)

    from pet_quantize.plugins.converters.vision_rknn_fp16 import VisionRknnFp16Converter

    plugin = VisionRknnFp16Converter(
        target_platform="rk3576",
        optimization_level=3,
        output_dir=str(tmp_path),
    )
    out = plugin.run(_make_vision_card(), recipe=MagicMock())

    assert calls["export"] == 1
    assert calls["convert"] == 1
    assert calls["onnx_received"] == str(onnx_out)

    assert len(out.edge_artifacts) == 1
    artifact = out.edge_artifacts[0]
    assert artifact.format == "rknn"
    assert artifact.target_hardware == ["rk3576"]
    assert artifact.artifact_uri.endswith(".rknn")
    assert len(artifact.sha256) == 64

    assert out.quantization is not None
    assert out.quantization.method == "fp16"
    assert out.intermediate_artifacts["vision_onnx_uri"] == str(onnx_out)


def test_converter_does_not_require_calibration(tmp_path, monkeypatch) -> None:
    """FP16 conversion does not need intermediate_artifacts.calibration_batch_uri."""
    onnx_out = tmp_path / "v.onnx"
    onnx_out.write_bytes(b"x")
    rknn_out = tmp_path / "v.rknn"
    rknn_out.write_bytes(b"y")

    from pet_quantize.convert import convert_to_rknn as _rknn_mod
    from pet_quantize.convert import export_vision_encoder as _export_mod

    monkeypatch.setattr(_export_mod, "export_vision_encoder", lambda config: str(onnx_out))
    monkeypatch.setattr(
        _rknn_mod, "convert_vision_to_rknn",
        lambda onnx_path, config, calib_dir=None: str(rknn_out),
    )

    from pet_quantize.plugins.converters.vision_rknn_fp16 import VisionRknnFp16Converter

    plugin = VisionRknnFp16Converter(output_dir=str(tmp_path))
    # ModelCard defaults intermediate_artifacts={} — no calibration_batch_uri present
    card = _make_vision_card()
    assert card.intermediate_artifacts == {}
    out = plugin.run(card, recipe=MagicMock())  # must not raise
    assert len(out.edge_artifacts) == 1
