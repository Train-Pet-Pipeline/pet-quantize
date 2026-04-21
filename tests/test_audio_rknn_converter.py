"""AudioRknnFp16Converter plugin tests."""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from pet_schema.model_card import ModelCard


def _make_audio_card() -> ModelCard:
    """Build a minimal valid ModelCard for audio conversion."""
    return ModelCard(
        id="audio-test",
        version="1.0.0",
        modality="audio",
        task="classification",
        arch="audio-cnn",
        training_recipe="test-recipe",
        hydra_config_sha="a" * 40,
        git_shas={"pet_quantize": "b" * 40},
        dataset_versions={"test": "v1"},
        checkpoint_uri="/tmp/audio-ckpt.pt",
        metrics={"accuracy": 0.9},
        gate_status="passed",
        trained_at=datetime.now(UTC),
        trained_by="test",
    )


def test_converter_registered() -> None:
    """Plugin registers under CONVERTERS['audio_rknn_fp16'] after module import."""
    from pet_infra.registry import CONVERTERS

    import pet_quantize.plugins.converters.audio_rknn_fp16  # noqa: F401

    assert "audio_rknn_fp16" in CONVERTERS.module_dict


def test_converter_calls_sdk_and_emits_edge_artifact(tmp_path, monkeypatch) -> None:
    """Happy path: SDK wrapper invoked, EdgeArtifact + QuantConfig set on card."""
    rknn_out = tmp_path / "audio.rknn"
    rknn_out.write_bytes(b"fake rknn contents")

    from pet_quantize.convert import convert_audio as _audio_mod

    calls = {"convert": 0}

    def fake_convert(config: dict) -> str:
        calls["convert"] += 1
        assert config["audio_checkpoint"] == "/tmp/audio-ckpt.pt"
        assert config["output_dir"] == str(tmp_path)
        return str(rknn_out)

    monkeypatch.setattr(_audio_mod, "convert_audio_to_rknn", fake_convert)

    from pet_quantize.plugins.converters.audio_rknn_fp16 import AudioRknnFp16Converter

    plugin = AudioRknnFp16Converter(
        target_platform="rk3576",
        output_dir=str(tmp_path),
    )
    out = plugin.run(_make_audio_card(), recipe=MagicMock())

    assert calls["convert"] == 1
    assert len(out.edge_artifacts) == 1
    artifact = out.edge_artifacts[0]
    assert artifact.format == "rknn"
    assert artifact.target_hardware == ["rk3576"]
    assert artifact.artifact_uri.endswith(".rknn")
    assert len(artifact.sha256) == 64
    assert artifact.input_shape == {"mel_spectrogram": [1, 1, 64, 100]}

    assert out.quantization is not None
    assert out.quantization.method == "fp16"


def test_converter_does_not_require_calibration(tmp_path, monkeypatch) -> None:
    """FP16 audio conversion does not need intermediate_artifacts.calibration_batch_uri."""
    rknn_out = tmp_path / "a.rknn"
    rknn_out.write_bytes(b"x")

    from pet_quantize.convert import convert_audio as _audio_mod

    monkeypatch.setattr(_audio_mod, "convert_audio_to_rknn", lambda config: str(rknn_out))

    from pet_quantize.plugins.converters.audio_rknn_fp16 import AudioRknnFp16Converter

    plugin = AudioRknnFp16Converter(output_dir=str(tmp_path))
    card = _make_audio_card()
    assert card.intermediate_artifacts == {}
    out = plugin.run(card, recipe=MagicMock())
    assert len(out.edge_artifacts) == 1
