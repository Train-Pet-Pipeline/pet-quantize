"""Combined VLM inference pipeline orchestrating ViT + LLM.

Runs quantized model inference and FP16 reference inference on the same
inputs, returning both for KL divergence comparison.
"""
from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pet_quantize.inference.rkllm_runner import RKLLMRunner
from pet_quantize.inference.rknn_runner import RKNNRunner

logger = logging.getLogger(__name__)


def _find_model_file(model_dir: str, pattern: str) -> str:
    """Find a model file matching a glob pattern in the model directory.

    Args:
        model_dir: Directory containing model files.
        pattern: Glob pattern to match.

    Returns:
        Path to the matched file.

    Raises:
        FileNotFoundError: If no matching file is found.
    """
    matches = glob.glob(str(Path(model_dir) / pattern))
    if not matches:
        msg = f"No file matching '{pattern}' in {model_dir}"
        raise FileNotFoundError(msg)
    return matches[0]


def _load_image(image_path: str, input_size: list[int]) -> np.ndarray:
    """Load and preprocess an image for vision encoder input.

    Args:
        image_path: Path to the image file.
        input_size: [height, width] for resizing.

    Returns:
        Numpy array of shape [1, 3, H, W].
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB").resize(
        (input_size[1], input_size[0])
    )
    arr = np.array(img, dtype=np.float32) / 255.0
    # HWC -> CHW -> NCHW
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)


def _build_prompt(image_path: str, schema_version: str) -> str:
    """Build the VLM prompt for a given image.

    Args:
        image_path: Path to the image being analyzed.
        schema_version: Schema version for prompt construction.

    Returns:
        Formatted prompt string.
    """
    from pet_schema import get_prompts

    prompts = get_prompts(schema_version)
    return f"{prompts['system']}\n{prompts['user']}"


def _run_fp16_reference(
    image_paths: list[str],
    fp16_weights_dir: str,
    schema_version: str,
) -> list[str]:
    """Run FP16 reference inference using transformers.

    Args:
        image_paths: List of image paths.
        fp16_weights_dir: Path to FP16 HuggingFace weights.
        schema_version: Schema version for prompt construction.

    Returns:
        List of JSON string outputs from the FP16 model.
    """
    if not fp16_weights_dir or not Path(fp16_weights_dir).exists():
        logger.warning("FP16 weights not available, skipping reference inference")
        return []

    from transformers import AutoModelForCausalLM, AutoProcessor

    logger.info("Loading FP16 reference model from %s", fp16_weights_dir)
    processor = AutoProcessor.from_pretrained(fp16_weights_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        fp16_weights_dir, trust_remote_code=True
    )
    model.eval()

    outputs: list[str] = []
    for img_path in image_paths:
        prompt = _build_prompt(img_path, schema_version)
        inputs = processor(text=prompt, images=img_path, return_tensors="pt")
        generated = model.generate(**inputs, max_new_tokens=2048)
        text = processor.decode(generated[0], skip_special_tokens=True)
        outputs.append(text)

    return outputs


def run_quantized_pipeline(
    model_dir: str,
    image_paths: list[str],
    device_id: str | None = None,
    params_path: str = "params.yaml",
) -> dict[str, Any]:
    """Run full VLM inference with quantized models.

    Orchestrates ViT (RKNN) + LLM (RKLLM) inference, and optionally
    runs FP16 reference inference for KL divergence comparison.

    Args:
        model_dir: Path to artifacts/converted/ directory.
        image_paths: List of image file paths to process.
        device_id: ADB device serial number, or None for simulated mode.
        params_path: Path to params.yaml.

    Returns:
        Dict with keys:
        - outputs (list[str]): Quantized model JSON outputs.
        - timings (list[float]): Per-image latency ms (empty if simulated).
        - fp16_outputs (list[str]): FP16 reference outputs for KL comparison.

    Raises:
        FileNotFoundError: If model_dir does not exist.
    """
    if not Path(model_dir).exists():
        msg = f"Model directory not found: {model_dir}"
        raise FileNotFoundError(msg)

    with open(params_path) as fh:
        params = yaml.safe_load(fh)

    inference_cfg = params.get("inference", {})
    convert_cfg = params.get("convert", {})
    schema_version = inference_cfg.get("schema_version", "1.0")
    input_size = convert_cfg.get("vision", {}).get("input_size", [448, 448])
    fp16_weights_dir = inference_cfg.get("fp16_weights_dir", "")

    is_device = device_id is not None
    target = convert_cfg.get("vision", {}).get("rknn_target") if is_device else None

    # Find model files
    rknn_path = _find_model_file(model_dir, "*.rknn")
    rkllm_path = _find_model_file(model_dir, "*.rkllm")

    # Initialize runners
    vision_runner = RKNNRunner(rknn_path, target=target, device_id=device_id)
    llm_runner = RKLLMRunner(rkllm_path, target=target, device_id=device_id)
    vision_runner.init()
    llm_runner.init()

    outputs: list[str] = []
    timings: list[float] = []

    try:
        for img_path in image_paths:
            # Vision encoding
            pixel_values = _load_image(img_path, input_size)
            visual_features, vis_time = vision_runner.infer([pixel_values])

            # LLM generation
            prompt = _build_prompt(img_path, schema_version)
            text, llm_time = llm_runner.generate(
                prompt=prompt,
                visual_features=visual_features[0],
            )
            outputs.append(text)

            if is_device:
                timings.append(vis_time + llm_time)
    finally:
        vision_runner.release()
        llm_runner.release()

    # FP16 reference outputs
    fp16_outputs = _run_fp16_reference(image_paths, fp16_weights_dir, schema_version)

    logger.info(
        "Pipeline complete: %d outputs, %d timings, %d fp16_outputs",
        len(outputs),
        len(timings),
        len(fp16_outputs),
    )

    return {
        "outputs": outputs,
        "timings": timings,
        "fp16_outputs": fp16_outputs,
    }
