"""Export vision encoder from HuggingFace model to ONNX format."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pet_infra.logging import setup_logging

logger = logging.getLogger(__name__)


def _load_model_auto(weights_dir: str) -> Any:
    """Load model using the appropriate class based on model type.

    Args:
        weights_dir: Path to the HuggingFace model directory.

    Returns:
        The loaded model instance.
    """
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(weights_dir, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")
    if model_type in ("qwen2_vl", "qwen2-vl", "qwen2_vl_text"):
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration.from_pretrained(
            weights_dir, trust_remote_code=True
        )
    return AutoModel.from_pretrained(weights_dir, trust_remote_code=True)


def export_vision_encoder(config: dict[str, Any]) -> str:
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
    import torch

    weights_dir = Path(config["weights_dir"])
    if not weights_dir.exists():
        raise FileNotFoundError(f"weights_dir not found: {weights_dir}")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    vision_cfg = config["vision"]
    h, w = vision_cfg["input_size"]
    opset = vision_cfg["onnx_opset"]

    logger.info("Loading model from %s", weights_dir)
    model = _load_model_auto(str(weights_dir))
    vision_model = model.visual
    vision_model.eval()

    output_path = output_dir / "vision_encoder.onnx"
    logger.info("Exporting vision encoder to %s (opset=%d)", output_path, opset)

    # Qwen2-VL uses NaViT-style dynamic patching: inputs are
    # (hidden_states: [num_patches, patch_dim], grid_thw: [num_images, 3])
    # rather than standard ViT (batch, 3, H, W).
    # Create dummy inputs matching a single image at the configured resolution.
    temporal, grid_h, grid_w = 1, h // 14, w // 14
    num_patches = temporal * grid_h * grid_w
    patch_dim = 14 * 14 * 3  # patch_size=14, channels=3
    dummy_hidden = torch.zeros(num_patches, patch_dim)
    dummy_grid = torch.tensor([[temporal, grid_h, grid_w]])

    # Use legacy ONNX exporter — the dynamo-based exporter in PyTorch 2.11+
    # cannot trace Qwen2-VL's rotary position embedding and dynamic attention.
    torch.onnx.export(
        vision_model,
        (dummy_hidden, dummy_grid),
        str(output_path),
        opset_version=max(opset, 18),
        input_names=["hidden_states", "grid_thw"],
        output_names=["vision_features"],
        dynamic_axes={
            "hidden_states": {0: "num_patches"},
            "grid_thw": {0: "num_images"},
            "vision_features": {0: "num_merged_patches"},
        },
        dynamo=False,
    )
    return str(output_path)


def main() -> None:
    """CLI entry point for vision encoder ONNX export."""
    setup_logging("pet-quantize")
    with open("params.yaml") as fh:
        params = yaml.safe_load(fh)
    result = export_vision_encoder(params["convert"])
    logger.info("Vision encoder exported", extra={"path": result})


if __name__ == "__main__":
    main()
