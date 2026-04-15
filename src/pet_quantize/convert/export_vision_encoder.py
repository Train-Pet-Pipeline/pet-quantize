"""Export vision encoder from HuggingFace model to ONNX format."""

import logging
from pathlib import Path

import torch
from transformers import AutoModel

logger = logging.getLogger(__name__)


def export_vision_encoder(config: dict) -> str:
    """Export the vision encoder sub-model to ONNX.

    Loads the HuggingFace model from config["weights_dir"], extracts
    model.visual, creates a dummy input from config["vision"]["input_size"],
    and exports to config["output_dir"]/vision_encoder.onnx.

    Args:
        config: Conversion config dict (the "convert" section of params.yaml).

    Returns:
        Absolute path to the exported ONNX file.

    Raises:
        FileNotFoundError: If weights_dir does not exist.
    """

    weights_dir = Path(config["weights_dir"])
    if not weights_dir.exists():
        raise FileNotFoundError(f"weights_dir not found: {weights_dir}")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    vision_cfg = config["vision"]
    h, w = vision_cfg["input_size"]
    opset = vision_cfg["onnx_opset"]

    logger.info("Loading model from %s", weights_dir)
    model = AutoModel.from_pretrained(str(weights_dir))
    vision_model = model.visual
    vision_model.eval()

    dummy_input = torch.zeros(1, 3, h, w)

    output_path = output_dir / "vision_encoder.onnx"
    logger.info("Exporting vision encoder to %s (opset=%d)", output_path, opset)
    torch.onnx.export(
        vision_model,
        (dummy_input,),
        str(output_path),
        opset_version=opset,
        input_names=["pixel_values"],
        output_names=["vision_features"],
    )
    return str(output_path)
