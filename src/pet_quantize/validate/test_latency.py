"""Smoke test: on-device inference latency."""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pytest


def _get_sample_images(params: dict, count: int) -> list[str]:
    """Get sample image paths for testing."""
    calib_dir = params["calibration"].get("output_dir", "artifacts/calibration")
    images = glob.glob(str(Path(calib_dir) / "*.jpg"))
    images += glob.glob(str(Path(calib_dir) / "*.png"))
    return images[:count]


def test_latency(params: dict, model_dir: str, device_id: str | None) -> None:
    """On-device P95 latency is within the threshold (4000ms)."""
    if device_id is None:
        pytest.skip("No device connected — latency test requires --device-id")

    from pet_quantize.inference import run_quantized_pipeline

    device_cfg = params["inference"]["device"]
    warmup_runs = device_cfg["warmup_runs"]
    latency_runs = device_cfg["latency_runs"]

    images = _get_sample_images(params, warmup_runs + latency_runs)
    if not images:
        pytest.skip("No calibration images available")

    # Warmup
    run_quantized_pipeline(
        model_dir=model_dir,
        image_paths=images[:warmup_runs],
        device_id=device_id,
        params_path="params.yaml",
    )

    # Measurement
    result = run_quantized_pipeline(
        model_dir=model_dir,
        image_paths=images[warmup_runs : warmup_runs + latency_runs],
        device_id=device_id,
        params_path="params.yaml",
    )

    timings = result["timings"]
    assert len(timings) > 0, "No timing data collected"

    p95 = float(np.percentile(timings, 95))
    threshold = params["gates"]["vlm"]["latency_p95_ms"]
    assert p95 <= threshold, f"P95 latency {p95:.0f}ms > {threshold}ms threshold"
