"""Convert ONNX model to RKNN format for RK3576 deployment."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def convert_vision_to_rknn(
    onnx_path: str, config: dict, calib_dir: str | None = None
) -> str:
    """Convert vision encoder ONNX model to RKNN FP16 format.

    Loads the ONNX model, builds an RKNN model with FP16 (no quantization),
    and exports to config["output_dir"]/vision_{target}.rknn.

    Args:
        onnx_path: Path to the input ONNX file.
        config: Conversion config dict (the "convert" section of params.yaml).
        calib_dir: Optional calibration dataset directory (unused for FP16).

    Returns:
        Absolute path to the exported RKNN file.

    Raises:
        FileNotFoundError: If onnx_path does not exist.
        RuntimeError: If RKNN load or build step fails.
    """
    from rknn.api import RKNN  # noqa: PLC0415

    onnx_file = Path(onnx_path)
    if not onnx_file.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_file}")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    vision_cfg = config["vision"]
    target = vision_cfg["rknn_target"]

    output_path = output_dir / f"vision_{target}.rknn"

    logger.info("Initialising RKNN for target=%s", target)
    rknn = RKNN()

    ret = rknn.load_onnx(model=str(onnx_file))
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

    return str(output_path)
