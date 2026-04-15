"""Inference interfaces for quantized models.

Public API:
    run_quantized_pipeline: Full VLM inference (vision + LLM).
    run_audio_inference: Audio CNN INT8 inference.
"""
from pet_quantize.inference.pipeline import run_quantized_pipeline
from pet_quantize.inference.rknn_runner import run_audio_inference

__all__ = ["run_quantized_pipeline", "run_audio_inference"]
