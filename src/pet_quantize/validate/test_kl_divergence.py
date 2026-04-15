"""Smoke test: KL divergence between FP16 and quantized outputs."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pytest


def _get_sample_images(params: dict, count: int) -> list[str]:
    """Get sample image paths for smoke testing."""
    calib_dir = params["calibration"].get("output_dir", "artifacts/calibration")
    images = glob.glob(str(Path(calib_dir) / "*.jpg"))
    images += glob.glob(str(Path(calib_dir) / "*.png"))
    return images[:count]


def _extract_distributions(output_json: str) -> list[float] | None:
    """Extract probability distributions from a model output JSON."""
    try:
        parsed = json.loads(output_json)
        dist = parsed.get("food_intake", {}).get("distribution", {})
        if dist:
            return list(dist.values())
    except (json.JSONDecodeError, AttributeError):
        pass
    return None


def _compute_kl(p: list[float], q: list[float]) -> float:
    """Compute KL divergence D(P || Q)."""
    p_arr = np.array(p, dtype=np.float64) + 1e-10
    q_arr = np.array(q, dtype=np.float64) + 1e-10
    p_arr /= p_arr.sum()
    q_arr /= q_arr.sum()
    return float(np.sum(p_arr * np.log(p_arr / q_arr)))


def test_kl_divergence(params: dict, model_dir: str, device_id: str | None) -> None:
    """KL divergence between FP16 and quantized outputs is within smoke threshold."""
    from pet_quantize.inference import run_quantized_pipeline

    sample_size = params["validate"]["smoke_sample_size"]
    kl_threshold = params["validate"]["kl_threshold"]
    images = _get_sample_images(params, sample_size)
    if not images:
        pytest.skip("No calibration images available for smoke testing")

    result = run_quantized_pipeline(
        model_dir=model_dir,
        image_paths=images,
        device_id=device_id,
        params_path="params.yaml",
    )

    if not result["fp16_outputs"]:
        pytest.skip("FP16 reference outputs not available")

    kl_values = []
    for q_out, fp16_out in zip(result["outputs"], result["fp16_outputs"]):
        q_dist = _extract_distributions(q_out)
        fp16_dist = _extract_distributions(fp16_out)
        if q_dist and fp16_dist and len(q_dist) == len(fp16_dist):
            kl_values.append(_compute_kl(fp16_dist, q_dist))

    if not kl_values:
        pytest.skip("No valid distribution pairs for KL comparison")

    mean_kl = float(np.mean(kl_values))
    assert mean_kl <= kl_threshold, (
        f"Mean KL divergence {mean_kl:.4f} > smoke threshold {kl_threshold}"
    )
