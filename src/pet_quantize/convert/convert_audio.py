"""Convert audio CNN PyTorch checkpoint to FP16 RKNN format for RK3576 deployment."""

import logging
import tempfile
from pathlib import Path

import torch
import yaml
from pet_infra.logging import setup_logging

logger = logging.getLogger(__name__)

# Audio model input shape: [batch, channels, mel_bins, time_frames]
_AUDIO_INPUT_SHAPE = (1, 1, 64, 100)


def convert_audio_to_rknn(config: dict) -> str:
    """Convert audio CNN checkpoint to RKNN FP16 format.

    Loads the PyTorch checkpoint, exports to a temporary ONNX file,
    then builds an RKNN model with FP16 (no quantization, no calibration
    required). The temporary ONNX file is cleaned up after conversion.

    Args:
        config: Conversion config dict (the "convert" section of params.yaml).

    Returns:
        Absolute path to the exported RKNN file.

    Raises:
        FileNotFoundError: If audio_checkpoint does not exist.
        RuntimeError: If RKNN load, build, or export step fails.
    """
    from rknn.api import RKNN  # noqa: PLC0415

    checkpoint_path = Path(config["audio_checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"audio_checkpoint not found: {checkpoint_path}")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "audio_cnn_fp16.rknn"

    logger.info("Loading audio CNN checkpoint from %s", checkpoint_path)
    model = torch.load(str(checkpoint_path), map_location="cpu")
    model.eval()

    dummy_input = torch.zeros(*_AUDIO_INPUT_SHAPE)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_f:
        tmp_onnx_path = tmp_f.name

    try:
        logger.info("Exporting audio CNN to temporary ONNX: %s", tmp_onnx_path)
        torch.onnx.export(
            model,
            (dummy_input,),
            tmp_onnx_path,
            opset_version=17,
            input_names=["mel_spectrogram"],
            output_names=["audio_features"],
        )

        logger.info("Initialising RKNN for audio CNN (FP16)")
        rknn = RKNN()

        ret = rknn.load_onnx(model=tmp_onnx_path)
        if ret != 0:
            raise RuntimeError(f"RKNN load_onnx failed with code {ret}")

        logger.info("Building RKNN model (FP16, no quantization)")
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"RKNN build failed with code {ret}")

        logger.info("Exporting RKNN model to %s", output_path)
        ret = rknn.export_rknn(str(output_path))
        if ret != 0:
            raise RuntimeError(f"RKNN export failed with code {ret}")

    finally:
        Path(tmp_onnx_path).unlink(missing_ok=True)
        logger.debug("Cleaned up temporary ONNX file: %s", tmp_onnx_path)

    return str(output_path)


def main() -> None:
    """CLI entry point for audio CNN RKNN conversion."""
    setup_logging("pet-quantize")
    with open("params.yaml") as fh:
        params = yaml.safe_load(fh)
    convert_cfg = params["convert"]
    result = convert_audio_to_rknn(convert_cfg)
    logger.info("Audio RKNN exported", extra={"path": result})


if __name__ == "__main__":
    main()
