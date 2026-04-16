"""Convert LLM weights to RKLLM W8A8 quantized format for RK3576 deployment."""

import logging
from pathlib import Path

import yaml
from pet_infra.logging import setup_logging

logger = logging.getLogger(__name__)


def convert_llm_to_rkllm(config: dict, calib_dir: str) -> str:
    """Convert Qwen2-VL LLM weights to RKLLM format with W8A8 quantization.

    Creates an RKLLMConverter with the specified model path, target platform,
    quantization scheme, and calibration data, then converts and exports the model.

    Args:
        config: Conversion config dict (the "convert" section of params.yaml).
        calib_dir: Path to calibration dataset directory.

    Returns:
        Absolute path to the exported RKLLM file.

    Raises:
        FileNotFoundError: If weights_dir does not exist.
        RuntimeError: If conversion or export fails.
    """
    from rkllm.api import RKLLMConverter  # noqa: PLC0415

    weights_dir = Path(config["weights_dir"])
    if not weights_dir.exists():
        raise FileNotFoundError(f"weights_dir not found: {weights_dir}")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_cfg = config["llm"]
    target = llm_cfg["rkllm_target"]
    quantization = llm_cfg["quantization"]

    output_path = output_dir / f"qwen2vl_2b_{quantization}_{target}.rkllm"

    logger.info(
        "Creating RKLLMConverter: target=%s quantization=%s", target, quantization
    )
    converter = RKLLMConverter(
        model_path=str(weights_dir),
        target_platform=target,
        quantization=quantization,
        calibration_data=calib_dir,
    )

    logger.info("Running RKLLM conversion")
    converter.convert()

    logger.info("Exporting RKLLM model to %s", output_path)
    converter.export(str(output_path))

    return str(output_path)


def main() -> None:
    """CLI entry point for RKLLM conversion."""
    setup_logging("pet-quantize")
    with open("params.yaml") as fh:
        params = yaml.safe_load(fh)
    convert_cfg = params["convert"]
    calib_dir = str(Path(params["calibration"]["output_dir"]))
    result = convert_llm_to_rkllm(convert_cfg, calib_dir)
    logger.info("RKLLM exported", extra={"path": result})


if __name__ == "__main__":
    main()
