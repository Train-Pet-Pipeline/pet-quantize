"""Smoke test: schema compliance of quantized model outputs."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest
from pet_schema import validate_output


def _get_sample_images(params: dict, count: int) -> list[str]:
    """Get sample image paths from calibration output for smoke testing."""
    calib_dir = params["calibration"].get("output_dir", "artifacts/calibration")
    images = glob.glob(str(Path(calib_dir) / "*.jpg"))
    images += glob.glob(str(Path(calib_dir) / "*.png"))
    return images[:count]


def test_schema_compliance(params: dict, model_dir: str, device_id: str | None) -> None:
    """Quantized model outputs pass pet-schema validation on a small sample."""
    from pet_quantize.inference import run_quantized_pipeline

    sample_size = params["validate"]["smoke_sample_size"]
    images = _get_sample_images(params, sample_size)
    if not images:
        pytest.skip("No calibration images available for smoke testing")

    result = run_quantized_pipeline(
        model_dir=model_dir,
        image_paths=images,
        device_id=device_id,
        params_path="params.yaml",
    )

    schema_version = params["inference"].get("schema_version", "1.0")
    total = len(result["outputs"])
    valid = 0
    for output in result["outputs"]:
        try:
            parsed = json.loads(output)
            validate_output(parsed, schema_version)
            valid += 1
        except Exception:
            pass

    compliance_rate = valid / total if total > 0 else 0
    assert compliance_rate >= 0.90, (
        f"Schema compliance {compliance_rate:.2%} < 90% smoke threshold"
    )
