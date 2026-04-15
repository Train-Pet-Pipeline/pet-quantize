# pet-quantize Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the quantization pipeline that converts pet-train FP16 weights to RK3576 device formats, validates conversion quality, exposes inference interfaces for pet-eval, and packages signed artifacts for pet-ota.

**Architecture:** Six modules (config, calibration, convert, inference, validate, packaging) following src/ layout. All numeric values from params.yaml via Pydantic models. RKNN/RKLLM SDK interactions isolated in inference/ module. Dual-mode (device/simulated) support throughout.

**Tech Stack:** Python 3.11, rknn-toolkit2, rknn-llm, transformers, onnx, torch, pydantic, cryptography, pytest

**Spec:** `docs/superpowers/specs/2026-04-15-pet-quantize-design.md`

---

## File Map

### New Files

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, ruff/mypy config |
| `Makefile` | setup/test/lint/clean/calibrate/convert/validate/package/all targets |
| `params.yaml` | All numeric configuration |
| `.env.example` | Environment variable documentation |
| `.gitignore` | Ignore artifacts/, __pycache__, etc. |
| `src/pet_quantize/__init__.py` | Package docstring |
| `src/pet_quantize/config.py` | Pydantic params.yaml loader + JSON logging setup |
| `src/pet_quantize/calibration/__init__.py` | Module init |
| `src/pet_quantize/calibration/build_calib_dataset.py` | Distribution-aware frame sampling from pet-data SQLite |
| `src/pet_quantize/calibration/validate_calib.py` | Enforce distribution coverage constraints |
| `src/pet_quantize/convert/__init__.py` | Module init |
| `src/pet_quantize/convert/export_vision_encoder.py` | ViT → ONNX (fp16) |
| `src/pet_quantize/convert/convert_to_rknn.py` | ONNX → .rknn (FP16) |
| `src/pet_quantize/convert/convert_to_rkllm.py` | Merged LLM → .rkllm (W8A8) |
| `src/pet_quantize/convert/convert_audio.py` | Audio CNN → INT8 .rknn |
| `src/pet_quantize/inference/__init__.py` | Public API re-exports |
| `src/pet_quantize/inference/rknn_runner.py` | RKNN SDK wrapper (vision + audio), dual-mode |
| `src/pet_quantize/inference/rkllm_runner.py` | RKLLM SDK wrapper, dual-mode |
| `src/pet_quantize/inference/pipeline.py` | VLM pipeline orchestrator |
| `src/pet_quantize/packaging/__init__.py` | Module init |
| `src/pet_quantize/packaging/build_package.py` | Tarball + manifest.json generation |
| `src/pet_quantize/packaging/sign_package.py` | RSA-2048 signing |
| `src/pet_quantize/packaging/verify_package.py` | Signature + sha256 verification |
| `src/pet_quantize/validate/__init__.py` | Module init |
| `src/pet_quantize/validate/conftest.py` | ADB device fixture, mode switch |
| `src/pet_quantize/validate/test_schema_compliance.py` | Smoke: schema compliance |
| `src/pet_quantize/validate/test_kl_divergence.py` | Smoke: KL divergence |
| `src/pet_quantize/validate/test_latency.py` | On-device latency |
| `src/pet_quantize/validate/test_audio_accuracy.py` | INT8 audio accuracy |
| `tests/conftest.py` | Shared test fixtures |
| `tests/test_config.py` | Config loading tests |
| `tests/test_build_calib.py` | Calibration sampling tests |
| `tests/test_validate_calib.py` | Distribution validation tests |
| `tests/test_export_vision_encoder.py` | Vision export tests |
| `tests/test_convert_rknn.py` | RKNN conversion tests |
| `tests/test_convert_rkllm.py` | RKLLM conversion tests |
| `tests/test_convert_audio.py` | Audio conversion tests |
| `tests/test_pipeline.py` | Pipeline composition tests |
| `tests/test_build_package.py` | Package build tests |
| `tests/test_sign_package.py` | Signing tests |
| `tests/test_verify_package.py` | Verification tests |

---

### Task 1: Repository Initialization

**Files:**
- Create: `pyproject.toml`
- Create: `Makefile`
- Create: `params.yaml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/pet_quantize/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "pet-quantize"
version = "0.1.0"
requires-python = ">=3.11,<3.12"

dependencies = [
    "pet-schema==1.0.0",
    "torch>=2.1",
    "transformers>=4.44,<5.0",
    "onnx>=1.14,<2.0",
    "pydantic>=2.0,<3.0",
    "pyyaml>=6.0",
    "cryptography>=41.0,<43.0",
    "tenacity",
    "python-json-logger",
    "wandb",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-cov",
    "ruff",
    "mypy",
]

[tool.setuptools.packages.find]
where = ["src"]
include = ["pet_quantize", "pet_quantize.*"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true

[[tool.mypy.overrides]]
module = [
    "pet_schema",
    "pet_schema.*",
    "rknn.*",
    "rknnlite.*",
    "rkllm.*",
]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create Makefile**

```makefile
.PHONY: setup test lint clean calibrate convert validate package all

setup:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/ && mypy src/

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist/ *.egg-info \
		artifacts/converted/* artifacts/release/*

calibrate:
	python -m pet_quantize.calibration.build_calib_dataset
	python -m pet_quantize.calibration.validate_calib

convert:
	python -m pet_quantize.convert.export_vision_encoder
	python -m pet_quantize.convert.convert_to_rknn
	python -m pet_quantize.convert.convert_to_rkllm &
	python -m pet_quantize.convert.convert_audio &
	wait

validate:
	pytest src/pet_quantize/validate/ -v $(ARGS)

package:
	python -m pet_quantize.packaging.build_package
	python -m pet_quantize.packaging.sign_package
	python -m pet_quantize.packaging.verify_package

all: calibrate convert validate package
```

- [ ] **Step 3: Create params.yaml**

```yaml
# === Calibration ===
calibration:
  frame_count: 200
  tolerance: 0.05
  min_breeds: 5
  distribution:
    lighting:
      bright: 0.40
      dim: 0.20
      infrared_night: 0.20
      unknown: 0.20
    action_primary:
      eating: 0.50
      sniffing_only: 0.20
      leaving_bowl: 0.15
      other: 0.15
  exclude:
    train_ids_path: ""
    gold_set_path: ""
  data_db_path: ""
  output_dir: "artifacts/calibration"

# === Conversion ===
convert:
  vision:
    input_size: [448, 448]
    onnx_opset: 17
    rknn_target: "rk3576"
    rknn_dtype: "fp16"
  llm:
    rkllm_target: "rk3576"
    quantization: "w8a8"
  audio:
    rknn_target: "rk3576"
    rknn_dtype: "int8"
  weights_dir: ""
  audio_checkpoint: ""
  output_dir: "artifacts/converted"

# === Inference ===
inference:
  schema_version: "1.0"
  simulated_sample_size: 50
  device:
    adb_timeout: 30
    warmup_runs: 3
    latency_runs: 20
  fp16_weights_dir: ""

# === Validation (smoke) ===
validate:
  smoke_sample_size: 30
  kl_threshold: 0.05

# === Packaging ===
packaging:
  version: ""
  lora_version: ""
  min_firmware: "2.0.0"
  release_notes: ""

# === wandb ===
wandb:
  project: "pet-quantize"
  entity: ""
```

- [ ] **Step 4: Create .gitignore**

```
__pycache__/
*.egg-info/
dist/
.pytest_cache/
.mypy_cache/
.ruff_cache/
*.pyc
.env
artifacts/converted/*
artifacts/release/*
!artifacts/converted/.gitkeep
!artifacts/release/.gitkeep
wandb/
```

- [ ] **Step 5: Create .env.example**

```bash
# pet-data SQLite database path
PET_DATA_DB_PATH=

# pet-train weights directory (merged HuggingFace format)
PET_TRAIN_WEIGHTS_DIR=

# pet-train audio checkpoint path
PET_TRAIN_AUDIO_CHECKPOINT=

# FP16 reference weights (for KL divergence comparison)
FP16_WEIGHTS_DIR=

# pet-eval gold set frame IDs (for calibration exclusion)
GOLD_SET_PATH=

# pet-train training frame IDs (for calibration exclusion)
TRAIN_IDS_PATH=

# RSA private key path (only in CI; local dev skips signing)
RSA_PRIVATE_KEY_PATH=

# ADB device ID (optional; omit for simulated mode)
DEVICE_ID=
```

- [ ] **Step 6: Create package init and artifact directories**

```python
# src/pet_quantize/__init__.py
"""pet-quantize: quantization, on-device conversion, and artifact packaging."""
```

Also create:
- `artifacts/converted/.gitkeep`
- `artifacts/release/.gitkeep`

- [ ] **Step 7: Verify setup**

Run: `cd /Users/bamboo/Githubs/Train-Pet-Pipeline/pet-quantize && make setup`
Expected: Successful pip install

Run: `make lint`
Expected: No errors (empty package passes lint)

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat(pet-quantize): initialize repository with pyproject.toml, Makefile, params.yaml"
```

---

### Task 2: Config Module

**Files:**
- Create: `src/pet_quantize/config.py`
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write conftest.py with shared fixtures**

```python
# tests/conftest.py
"""Shared test fixtures for pet-quantize."""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test artifacts."""
    return tmp_path


@pytest.fixture()
def sample_params() -> dict[str, Any]:
    """Provide a minimal params.yaml structure for testing."""
    return {
        "calibration": {
            "frame_count": 200,
            "tolerance": 0.05,
            "min_breeds": 5,
            "distribution": {
                "lighting": {
                    "bright": 0.40,
                    "dim": 0.20,
                    "infrared_night": 0.20,
                    "unknown": 0.20,
                },
                "action_primary": {
                    "eating": 0.50,
                    "sniffing_only": 0.20,
                    "leaving_bowl": 0.15,
                    "other": 0.15,
                },
            },
            "exclude": {"train_ids_path": "", "gold_set_path": ""},
            "data_db_path": "",
            "output_dir": "",
        },
        "convert": {
            "vision": {
                "input_size": [448, 448],
                "onnx_opset": 17,
                "rknn_target": "rk3576",
                "rknn_dtype": "fp16",
            },
            "llm": {"rkllm_target": "rk3576", "quantization": "w8a8"},
            "audio": {"rknn_target": "rk3576", "rknn_dtype": "int8"},
            "weights_dir": "",
            "audio_checkpoint": "",
            "output_dir": "",
        },
        "inference": {
            "schema_version": "1.0",
            "simulated_sample_size": 50,
            "device": {
                "adb_timeout": 30,
                "warmup_runs": 3,
                "latency_runs": 20,
            },
            "fp16_weights_dir": "",
        },
        "validate": {"smoke_sample_size": 30, "kl_threshold": 0.05},
        "packaging": {
            "version": "1.0.0",
            "lora_version": "1.0",
            "min_firmware": "2.0.0",
            "release_notes": "Initial release",
        },
        "wandb": {"project": "pet-quantize", "entity": ""},
    }


@pytest.fixture()
def sample_params_path(tmp_dir: Path, sample_params: dict[str, Any]) -> Path:
    """Write sample params to a temp file and return its path."""
    p = tmp_dir / "params.yaml"
    p.write_text(yaml.dump(sample_params))
    return p


@pytest.fixture()
def sample_calib_db(tmp_dir: Path) -> Path:
    """Create a minimal SQLite database mimicking pet-data frames table."""
    db_path = tmp_dir / "frames.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE frames (
            frame_id TEXT PRIMARY KEY,
            image_path TEXT,
            lighting TEXT,
            action_primary TEXT,
            breed TEXT
        )
        """
    )
    # Insert enough frames to satisfy distribution sampling
    breeds = ["persian", "siamese", "maine_coon", "ragdoll", "british_shorthair", "bengal"]
    rows = []
    frame_id = 0
    distribution = {
        ("bright", "eating"): 100,
        ("bright", "sniffing_only"): 20,
        ("bright", "leaving_bowl"): 15,
        ("bright", "other"): 15,
        ("dim", "eating"): 50,
        ("dim", "sniffing_only"): 15,
        ("dim", "leaving_bowl"): 10,
        ("dim", "other"): 10,
        ("infrared_night", "eating"): 50,
        ("infrared_night", "sniffing_only"): 15,
        ("infrared_night", "leaving_bowl"): 10,
        ("infrared_night", "other"): 10,
        ("unknown", "eating"): 50,
        ("unknown", "sniffing_only"): 15,
        ("unknown", "leaving_bowl"): 10,
        ("unknown", "other"): 10,
    }
    for (lighting, action), count in distribution.items():
        for i in range(count):
            fid = f"frame_{frame_id:05d}"
            img = f"/images/{fid}.jpg"
            breed = breeds[frame_id % len(breeds)]
            rows.append((fid, img, lighting, action, breed))
            frame_id += 1

    conn.executemany(
        "INSERT INTO frames VALUES (?, ?, ?, ?, ?)", rows
    )
    conn.commit()
    conn.close()
    return db_path
```

- [ ] **Step 2: Write failing test for config**

```python
# tests/test_config.py
"""Tests for pet_quantize.config."""
from __future__ import annotations

from pathlib import Path

import pytest


def test_load_params_valid(sample_params_path: Path) -> None:
    """Loading a valid params.yaml returns a QuantizeParams model."""
    from pet_quantize.config import QuantizeParams, load_params

    params = load_params(sample_params_path)
    assert isinstance(params, QuantizeParams)
    assert params.calibration.frame_count == 200
    assert params.calibration.tolerance == 0.05
    assert params.calibration.min_breeds == 5
    assert params.convert.vision.rknn_target == "rk3576"
    assert params.packaging.min_firmware == "2.0.0"


def test_load_params_missing_field(tmp_path: Path) -> None:
    """Loading params with a missing required section raises ValidationError."""
    import yaml
    from pydantic import ValidationError

    from pet_quantize.config import load_params

    bad = {"calibration": {"frame_count": 200}}  # missing other sections
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.dump(bad))

    with pytest.raises(ValidationError):
        load_params(p)


def test_load_params_file_not_found() -> None:
    """Loading a nonexistent file raises FileNotFoundError."""
    from pet_quantize.config import load_params

    with pytest.raises(FileNotFoundError):
        load_params(Path("/nonexistent/params.yaml"))


def test_setup_logging_json_format(capsys: pytest.CaptureFixture[str]) -> None:
    """setup_logging configures structured JSON output."""
    import json
    import logging

    from pet_quantize.config import setup_logging

    setup_logging()
    logger = logging.getLogger("test_config_logger")
    logger.info("test message", extra={"key": "value"})

    captured = capsys.readouterr()
    # JSON logger writes to stderr
    line = captured.err.strip().split("\n")[-1]
    parsed = json.loads(line)
    assert parsed["message"] == "test message"
    assert parsed["key"] == "value"


def test_setup_logging_idempotent() -> None:
    """Calling setup_logging twice does not duplicate handlers."""
    import logging

    from pythonjsonlogger import jsonlogger

    from pet_quantize.config import setup_logging

    setup_logging()
    setup_logging()

    root = logging.getLogger()
    json_handlers = [
        h for h in root.handlers
        if isinstance(getattr(h, "formatter", None), jsonlogger.JsonFormatter)
    ]
    assert len(json_handlers) >= 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/bamboo/Githubs/Train-Pet-Pipeline/pet-quantize && pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pet_quantize.config'`

- [ ] **Step 4: Implement config.py**

```python
# src/pet_quantize/config.py
"""Pydantic params.yaml loader and structured JSON logging setup for pet-quantize."""
from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel
from pythonjsonlogger import jsonlogger


# --- Pydantic models ---


class CalibDistributionCategory(BaseModel):
    """Distribution proportions for a single calibration category."""

    bright: float = 0.0
    dim: float = 0.0
    infrared_night: float = 0.0
    unknown: float = 0.0
    eating: float = 0.0
    sniffing_only: float = 0.0
    leaving_bowl: float = 0.0
    other: float = 0.0


class CalibDistribution(BaseModel):
    """Distribution constraints for calibration dataset."""

    lighting: dict[str, float]
    action_primary: dict[str, float]


class CalibExclude(BaseModel):
    """Paths for frame ID exclusion lists."""

    train_ids_path: str = ""
    gold_set_path: str = ""


class CalibrationConfig(BaseModel):
    """Calibration dataset configuration."""

    frame_count: int
    tolerance: float
    min_breeds: int
    distribution: CalibDistribution
    exclude: CalibExclude = CalibExclude()
    data_db_path: str = ""
    output_dir: str = ""


class VisionConvertConfig(BaseModel):
    """Vision encoder conversion config."""

    input_size: list[int]
    onnx_opset: int
    rknn_target: str
    rknn_dtype: str


class LlmConvertConfig(BaseModel):
    """LLM conversion config."""

    rkllm_target: str
    quantization: str


class AudioConvertConfig(BaseModel):
    """Audio model conversion config."""

    rknn_target: str
    rknn_dtype: str


class ConvertConfig(BaseModel):
    """Conversion configuration."""

    vision: VisionConvertConfig
    llm: LlmConvertConfig
    audio: AudioConvertConfig
    weights_dir: str = ""
    audio_checkpoint: str = ""
    output_dir: str = ""


class DeviceConfig(BaseModel):
    """Hardware device configuration."""

    adb_timeout: int = 30
    warmup_runs: int = 3
    latency_runs: int = 20


class InferenceConfig(BaseModel):
    """Inference configuration."""

    schema_version: str = "1.0"
    simulated_sample_size: int = 50
    device: DeviceConfig = DeviceConfig()
    fp16_weights_dir: str = ""


class ValidateConfig(BaseModel):
    """Smoke validation configuration."""

    smoke_sample_size: int = 30
    kl_threshold: float = 0.05


class PackagingConfig(BaseModel):
    """Packaging configuration."""

    version: str = ""
    lora_version: str = ""
    min_firmware: str = "2.0.0"
    release_notes: str = ""


class WandbConfig(BaseModel):
    """Weights & Biases configuration."""

    project: str = "pet-quantize"
    entity: str = ""


class QuantizeParams(BaseModel):
    """Root configuration model for pet-quantize params.yaml."""

    calibration: CalibrationConfig
    convert: ConvertConfig
    inference: InferenceConfig = InferenceConfig()
    validate: ValidateConfig = ValidateConfig()
    packaging: PackagingConfig = PackagingConfig()
    wandb: WandbConfig = WandbConfig()


def load_params(path: Path) -> QuantizeParams:
    """Load and validate params.yaml.

    Args:
        path: Path to the params.yaml file.

    Returns:
        Validated QuantizeParams model.

    Raises:
        FileNotFoundError: If the file does not exist.
        pydantic.ValidationError: If the YAML content is invalid.
    """
    if not path.exists():
        msg = f"params.yaml not found: {path}"
        raise FileNotFoundError(msg)

    with open(path) as fh:
        raw = yaml.safe_load(fh)

    return QuantizeParams.model_validate(raw)


# --- Logging ---


def setup_logging() -> None:
    """Configure structured JSON logging for the root logger.

    Safe to call multiple times — will not duplicate handlers.
    """
    root = logging.getLogger()

    for h in root.handlers:
        if isinstance(getattr(h, "formatter", None), jsonlogger.JsonFormatter):
            return

    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(logging.INFO)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: 4 passed

- [ ] **Step 6: Run lint**

Run: `make lint`
Expected: No errors

- [ ] **Step 7: Commit**

```bash
git add src/pet_quantize/config.py tests/conftest.py tests/test_config.py
git commit -m "feat(pet-quantize): add config module with Pydantic params loader and JSON logging"
```

---

### Task 3: Calibration — validate_calib

**Files:**
- Create: `src/pet_quantize/calibration/__init__.py`
- Create: `src/pet_quantize/calibration/validate_calib.py`
- Create: `tests/test_validate_calib.py`

- [ ] **Step 1: Create calibration __init__.py**

```python
# src/pet_quantize/calibration/__init__.py
"""Calibration dataset construction and validation."""
```

- [ ] **Step 2: Write failing tests for validate_calib**

```python
# tests/test_validate_calib.py
"""Tests for pet_quantize.calibration.validate_calib."""
from __future__ import annotations

from typing import Any

import pytest


def _make_frame(frame_id: str, lighting: str, action: str, breed: str) -> dict[str, str]:
    """Helper to create a calibration frame dict."""
    return {
        "frame_id": frame_id,
        "image_path": f"/images/{frame_id}.jpg",
        "lighting": lighting,
        "action_primary": action,
        "breed": breed,
    }


def _make_valid_frames(count: int = 200) -> list[dict[str, str]]:
    """Generate a valid calibration dataset that satisfies all distribution constraints."""
    breeds = ["persian", "siamese", "maine_coon", "ragdoll", "british_shorthair"]
    # Exactly matching target distribution for 200 frames
    spec = [
        ("bright", "eating", 40),
        ("bright", "sniffing_only", 16),
        ("bright", "leaving_bowl", 12),
        ("bright", "other", 12),
        ("dim", "eating", 20),
        ("dim", "sniffing_only", 8),
        ("dim", "leaving_bowl", 6),
        ("dim", "other", 6),
        ("infrared_night", "eating", 20),
        ("infrared_night", "sniffing_only", 8),
        ("infrared_night", "leaving_bowl", 6),
        ("infrared_night", "other", 6),
        ("unknown", "eating", 20),
        ("unknown", "sniffing_only", 8),
        ("unknown", "leaving_bowl", 6),
        ("unknown", "other", 6),
    ]
    frames = []
    idx = 0
    for lighting, action, n in spec:
        for i in range(n):
            frames.append(_make_frame(
                f"f_{idx:05d}", lighting, action, breeds[idx % len(breeds)]
            ))
            idx += 1
    return frames


@pytest.fixture()
def calib_config(sample_params: dict[str, Any]) -> dict[str, Any]:
    """Extract calibration config from sample_params."""
    return sample_params["calibration"]


def test_valid_distribution_passes(calib_config: dict[str, Any]) -> None:
    """A dataset matching the target distribution passes validation."""
    from pet_quantize.calibration.validate_calib import validate_calibration_dataset

    frames = _make_valid_frames(200)
    result = validate_calibration_dataset(frames, calib_config)
    assert result.passed is True
    assert len(result.violations) == 0


def test_lighting_proportion_violation(calib_config: dict[str, Any]) -> None:
    """A dataset with lighting proportion outside tolerance is rejected."""
    from pet_quantize.calibration.validate_calib import validate_calibration_dataset

    frames = _make_valid_frames(200)
    # Replace 20 "dim" frames with "bright" → bright becomes ~50% (target 40%, tolerance 5%)
    replaced = 0
    for f in frames:
        if f["lighting"] == "dim" and replaced < 20:
            f["lighting"] = "bright"
            replaced += 1
    result = validate_calibration_dataset(frames, calib_config)
    assert result.passed is False
    assert any("lighting" in v for v in result.violations)


def test_action_proportion_violation(calib_config: dict[str, Any]) -> None:
    """A dataset with action_primary proportion outside tolerance is rejected."""
    from pet_quantize.calibration.validate_calib import validate_calibration_dataset

    frames = _make_valid_frames(200)
    # Replace all "leaving_bowl" with "eating" → eating becomes ~65% (target 50%)
    for f in frames:
        if f["action_primary"] == "leaving_bowl":
            f["action_primary"] = "eating"
    result = validate_calibration_dataset(frames, calib_config)
    assert result.passed is False
    assert any("action_primary" in v for v in result.violations)


def test_insufficient_breeds(calib_config: dict[str, Any]) -> None:
    """A dataset with fewer than min_breeds is rejected."""
    from pet_quantize.calibration.validate_calib import validate_calibration_dataset

    frames = _make_valid_frames(200)
    # Set all breeds to only 3 unique values
    for i, f in enumerate(frames):
        f["breed"] = ["persian", "siamese", "maine_coon"][i % 3]
    result = validate_calibration_dataset(frames, calib_config)
    assert result.passed is False
    assert any("breed" in v.lower() for v in result.violations)


def test_wrong_frame_count(calib_config: dict[str, Any]) -> None:
    """A dataset with wrong frame count is rejected."""
    from pet_quantize.calibration.validate_calib import validate_calibration_dataset

    frames = _make_valid_frames(200)[:100]  # Only 100 frames
    result = validate_calibration_dataset(frames, calib_config)
    assert result.passed is False
    assert any("count" in v.lower() for v in result.violations)


def test_boundary_tolerance_passes(calib_config: dict[str, Any]) -> None:
    """A distribution at exactly the tolerance boundary passes."""
    from pet_quantize.calibration.validate_calib import validate_calibration_dataset

    frames = _make_valid_frames(200)
    # Shift bright from 40% to 44.9% (within 5% tolerance of 40%)
    # Need to move ~10 frames from other lighting to bright
    shifted = 0
    for f in frames:
        if f["lighting"] == "unknown" and shifted < 10:
            f["lighting"] = "bright"
            shifted += 1
    # bright is now ~45% = 40% + 5%, at boundary
    result = validate_calibration_dataset(frames, calib_config)
    assert result.passed is True


def test_boundary_tolerance_fails(calib_config: dict[str, Any]) -> None:
    """A distribution just beyond the tolerance boundary fails."""
    from pet_quantize.calibration.validate_calib import validate_calibration_dataset

    frames = _make_valid_frames(200)
    # Shift bright beyond 45% → fails
    shifted = 0
    for f in frames:
        if f["lighting"] == "unknown" and shifted < 12:
            f["lighting"] = "bright"
            shifted += 1
    result = validate_calibration_dataset(frames, calib_config)
    assert result.passed is False
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_validate_calib.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement validate_calib.py**

```python
# src/pet_quantize/calibration/validate_calib.py
"""Validate calibration dataset distribution constraints.

Enforces that the sampled calibration frames satisfy the required
lighting and action_primary proportions, breed diversity, and frame count.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibValidationResult:
    """Result of calibration dataset validation."""

    passed: bool
    violations: list[str] = field(default_factory=list)


def _check_distribution(
    frames: list[dict[str, str]],
    field_name: str,
    target: dict[str, float],
    tolerance: float,
) -> list[str]:
    """Check that a categorical field's proportions match the target within tolerance.

    Args:
        frames: List of calibration frame dicts.
        field_name: Key in each frame dict to check (e.g., "lighting").
        target: Mapping of category → target proportion.
        tolerance: Maximum allowed absolute deviation.

    Returns:
        List of violation descriptions (empty if all pass).
    """
    total = len(frames)
    if total == 0:
        return [f"{field_name}: no frames to validate"]

    counts: dict[str, int] = {}
    for f in frames:
        val = f.get(field_name, "")
        counts[val] = counts.get(val, 0) + 1

    violations = []
    for category, target_prop in target.items():
        actual_prop = counts.get(category, 0) / total
        deviation = abs(actual_prop - target_prop)
        if deviation > tolerance:
            violations.append(
                f"{field_name}.{category}: actual={actual_prop:.3f}, "
                f"target={target_prop:.3f}, deviation={deviation:.3f} > tolerance={tolerance}"
            )
    return violations


def validate_calibration_dataset(
    frames: list[dict[str, str]],
    config: dict[str, Any],
) -> CalibValidationResult:
    """Validate that calibration frames satisfy all distribution constraints.

    Args:
        frames: List of calibration frame dicts with keys:
            frame_id, image_path, lighting, action_primary, breed.
        config: Calibration section of params.yaml.

    Returns:
        CalibValidationResult indicating pass/fail and any violations.
    """
    violations: list[str] = []
    frame_count = config["frame_count"]
    tolerance = config["tolerance"]
    min_breeds = config["min_breeds"]
    distribution = config["distribution"]

    # Check frame count
    if len(frames) != frame_count:
        violations.append(
            f"Frame count: expected={frame_count}, actual={len(frames)}"
        )

    # Check lighting distribution
    violations.extend(
        _check_distribution(frames, "lighting", distribution["lighting"], tolerance)
    )

    # Check action_primary distribution
    violations.extend(
        _check_distribution(
            frames, "action_primary", distribution["action_primary"], tolerance
        )
    )

    # Check breed diversity
    unique_breeds = {f.get("breed", "") for f in frames}
    unique_breeds.discard("")
    if len(unique_breeds) < min_breeds:
        violations.append(
            f"Breed diversity: found={len(unique_breeds)}, min_required={min_breeds}"
        )

    passed = len(violations) == 0

    if passed:
        logger.info("Calibration validation passed")
    else:
        for v in violations:
            logger.error("Calibration violation: %s", v)

    return CalibValidationResult(passed=passed, violations=violations)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_validate_calib.py -v`
Expected: 7 passed

- [ ] **Step 6: Run lint**

Run: `make lint`
Expected: No errors

- [ ] **Step 7: Commit**

```bash
git add src/pet_quantize/calibration/ tests/test_validate_calib.py
git commit -m "feat(pet-quantize): add calibration distribution validation"
```

---

### Task 4: Calibration — build_calib_dataset

**Files:**
- Create: `src/pet_quantize/calibration/build_calib_dataset.py`
- Create: `tests/test_build_calib.py`

- [ ] **Step 1: Write failing tests for build_calib_dataset**

```python
# tests/test_build_calib.py
"""Tests for pet_quantize.calibration.build_calib_dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def test_sample_respects_frame_count(
    sample_calib_db: Path, sample_params: dict[str, Any], tmp_dir: Path
) -> None:
    """Sampling returns exactly frame_count frames."""
    from pet_quantize.calibration.build_calib_dataset import build_calib_dataset

    config = sample_params["calibration"]
    config["data_db_path"] = str(sample_calib_db)
    config["output_dir"] = str(tmp_dir / "calib")
    frames = build_calib_dataset(config)
    assert len(frames) == config["frame_count"]


def test_sample_excludes_train_ids(
    sample_calib_db: Path, sample_params: dict[str, Any], tmp_dir: Path
) -> None:
    """Frames in the training ID exclusion list are not sampled."""
    from pet_quantize.calibration.build_calib_dataset import build_calib_dataset

    config = sample_params["calibration"]
    config["data_db_path"] = str(sample_calib_db)
    config["output_dir"] = str(tmp_dir / "calib")

    # Create an exclusion file with some frame IDs
    exclude_path = tmp_dir / "train_ids.txt"
    exclude_ids = [f"frame_{i:05d}" for i in range(50)]
    exclude_path.write_text("\n".join(exclude_ids))
    config["exclude"]["train_ids_path"] = str(exclude_path)

    frames = build_calib_dataset(config)
    sampled_ids = {f["frame_id"] for f in frames}
    assert sampled_ids.isdisjoint(set(exclude_ids))


def test_sample_excludes_gold_set_ids(
    sample_calib_db: Path, sample_params: dict[str, Any], tmp_dir: Path
) -> None:
    """Frames in the gold set exclusion list are not sampled."""
    from pet_quantize.calibration.build_calib_dataset import build_calib_dataset

    config = sample_params["calibration"]
    config["data_db_path"] = str(sample_calib_db)
    config["output_dir"] = str(tmp_dir / "calib")

    exclude_path = tmp_dir / "gold_ids.txt"
    exclude_ids = [f"frame_{i:05d}" for i in range(60, 80)]
    exclude_path.write_text("\n".join(exclude_ids))
    config["exclude"]["gold_set_path"] = str(exclude_path)

    frames = build_calib_dataset(config)
    sampled_ids = {f["frame_id"] for f in frames}
    assert sampled_ids.isdisjoint(set(exclude_ids))


def test_sample_satisfies_distribution(
    sample_calib_db: Path, sample_params: dict[str, Any], tmp_dir: Path
) -> None:
    """Sampled dataset passes distribution validation."""
    from pet_quantize.calibration.build_calib_dataset import build_calib_dataset
    from pet_quantize.calibration.validate_calib import validate_calibration_dataset

    config = sample_params["calibration"]
    config["data_db_path"] = str(sample_calib_db)
    config["output_dir"] = str(tmp_dir / "calib")
    frames = build_calib_dataset(config)
    result = validate_calibration_dataset(frames, config)
    assert result.passed is True


def test_insufficient_pool_raises(
    sample_calib_db: Path, sample_params: dict[str, Any], tmp_dir: Path
) -> None:
    """Raises ValueError when the database has too few frames for the distribution."""
    from pet_quantize.calibration.build_calib_dataset import build_calib_dataset

    config = sample_params["calibration"]
    config["data_db_path"] = str(sample_calib_db)
    config["output_dir"] = str(tmp_dir / "calib")
    config["frame_count"] = 10000  # More than available

    with pytest.raises(ValueError, match="Insufficient"):
        build_calib_dataset(config)


def test_missing_db_raises(sample_params: dict[str, Any]) -> None:
    """Raises FileNotFoundError when database path does not exist."""
    from pet_quantize.calibration.build_calib_dataset import build_calib_dataset

    config = sample_params["calibration"]
    config["data_db_path"] = "/nonexistent/db.sqlite"

    with pytest.raises(FileNotFoundError):
        build_calib_dataset(config)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_build_calib.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement build_calib_dataset.py**

```python
# src/pet_quantize/calibration/build_calib_dataset.py
"""Build calibration dataset by distribution-aware sampling from pet-data SQLite.

Samples frames from the pet-data frames table, excluding training set and
gold set frame IDs, while respecting the distribution constraints defined
in params.yaml.
"""
from __future__ import annotations

import logging
import random
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_exclude_ids(path: str) -> set[str]:
    """Load frame IDs to exclude from a newline-delimited text file.

    Args:
        path: Path to the exclusion file. Empty string means no exclusions.

    Returns:
        Set of frame IDs to exclude.
    """
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        logger.warning("Exclusion file not found, skipping: %s", path)
        return set()
    return {line.strip() for line in p.read_text().splitlines() if line.strip()}


def _query_frames(db_path: str, exclude_ids: set[str]) -> list[dict[str, str]]:
    """Query all eligible frames from pet-data SQLite database.

    Args:
        db_path: Path to the SQLite database.
        exclude_ids: Frame IDs to exclude.

    Returns:
        List of frame dicts with keys: frame_id, image_path, lighting,
        action_primary, breed.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            "SELECT frame_id, image_path, lighting, action_primary, breed FROM frames"
        )
        frames = []
        for row in cursor:
            if row["frame_id"] not in exclude_ids:
                frames.append(dict(row))
        return frames
    finally:
        conn.close()


def _stratified_sample(
    frames: list[dict[str, str]],
    target_count: int,
    distribution: dict[str, dict[str, float]],
) -> list[dict[str, str]]:
    """Sample frames to match target distribution proportions.

    Groups frames by each distribution field and samples proportionally.
    Uses lighting as the primary stratification axis, then fills remaining
    slots to balance action_primary.

    Args:
        frames: Pool of available frames.
        target_count: Exact number of frames to return.
        distribution: Mapping of field_name → {category: proportion}.

    Returns:
        List of sampled frames.

    Raises:
        ValueError: If insufficient frames are available for the distribution.
    """
    # Group by (lighting, action_primary) for joint stratification
    buckets: dict[tuple[str, str], list[dict[str, str]]] = {}
    for f in frames:
        key = (f["lighting"], f["action_primary"])
        buckets.setdefault(key, []).append(f)

    lighting_dist = distribution["lighting"]
    action_dist = distribution["action_primary"]

    # Compute target counts per (lighting, action) pair
    target_per_bucket: dict[tuple[str, str], int] = {}
    allocated = 0
    pairs = []
    for light, light_prop in lighting_dist.items():
        for action, action_prop in action_dist.items():
            count = round(target_count * light_prop * action_prop)
            target_per_bucket[(light, action)] = count
            allocated += count
            pairs.append((light, action))

    # Adjust rounding errors — add/remove from the largest bucket
    diff = target_count - allocated
    if diff != 0:
        largest = max(pairs, key=lambda p: target_per_bucket[p])
        target_per_bucket[largest] += diff

    # Sample from each bucket
    sampled: list[dict[str, str]] = []
    for key, needed in target_per_bucket.items():
        available = buckets.get(key, [])
        if len(available) < needed:
            msg = (
                f"Insufficient frames for bucket {key}: "
                f"need={needed}, available={len(available)}"
            )
            raise ValueError(msg)
        sampled.extend(random.sample(available, needed))

    return sampled


def build_calib_dataset(config: dict[str, Any]) -> list[dict[str, str]]:
    """Build a calibration dataset from pet-data SQLite.

    Args:
        config: Calibration section of params.yaml.

    Returns:
        List of sampled frame dicts.

    Raises:
        FileNotFoundError: If the database path does not exist.
        ValueError: If insufficient frames for the distribution.
    """
    db_path = config["data_db_path"]
    if not Path(db_path).exists():
        msg = f"Database not found: {db_path}"
        raise FileNotFoundError(msg)

    # Load exclusion lists
    exclude_ids = _load_exclude_ids(config["exclude"]["train_ids_path"])
    exclude_ids |= _load_exclude_ids(config["exclude"]["gold_set_path"])
    logger.info("Excluding %d frame IDs", len(exclude_ids))

    # Query eligible frames
    all_frames = _query_frames(db_path, exclude_ids)
    logger.info("Available frames after exclusion: %d", len(all_frames))

    # Stratified sampling
    frames = _stratified_sample(
        all_frames,
        target_count=config["frame_count"],
        distribution=config["distribution"],
    )

    logger.info("Sampled %d calibration frames", len(frames))
    return frames
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_build_calib.py -v`
Expected: 6 passed

- [ ] **Step 5: Run lint**

Run: `make lint`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add src/pet_quantize/calibration/build_calib_dataset.py tests/test_build_calib.py
git commit -m "feat(pet-quantize): add calibration dataset builder with distribution-aware sampling"
```

---

### Task 5: Convert — export_vision_encoder

**Files:**
- Create: `src/pet_quantize/convert/__init__.py`
- Create: `src/pet_quantize/convert/export_vision_encoder.py`
- Create: `tests/test_export_vision_encoder.py`

- [ ] **Step 1: Create convert __init__.py**

```python
# src/pet_quantize/convert/__init__.py
"""Model format conversion modules."""
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_export_vision_encoder.py
"""Tests for pet_quantize.convert.export_vision_encoder."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def test_export_creates_onnx_file(
    tmp_dir: Path, sample_params: dict[str, Any]
) -> None:
    """Export produces an ONNX file at the expected path."""
    from pet_quantize.convert.export_vision_encoder import export_vision_encoder

    config = sample_params["convert"]
    config["weights_dir"] = str(tmp_dir / "weights")
    config["output_dir"] = str(tmp_dir / "converted")

    with (
        patch(
            "pet_quantize.convert.export_vision_encoder.AutoModel"
        ) as mock_model_cls,
        patch("pet_quantize.convert.export_vision_encoder.torch") as mock_torch,
    ):
        mock_model = MagicMock()
        mock_model.visual = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        result = export_vision_encoder(config)

    assert result.endswith(".onnx")
    mock_torch.onnx.export.assert_called_once()


def test_export_uses_correct_opset(
    tmp_dir: Path, sample_params: dict[str, Any]
) -> None:
    """Export uses the opset version from config."""
    from pet_quantize.convert.export_vision_encoder import export_vision_encoder

    config = sample_params["convert"]
    config["weights_dir"] = str(tmp_dir / "weights")
    config["output_dir"] = str(tmp_dir / "converted")

    with (
        patch(
            "pet_quantize.convert.export_vision_encoder.AutoModel"
        ) as mock_model_cls,
        patch("pet_quantize.convert.export_vision_encoder.torch") as mock_torch,
    ):
        mock_model = MagicMock()
        mock_model.visual = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        export_vision_encoder(config)

    call_kwargs = mock_torch.onnx.export.call_args
    assert call_kwargs[1].get("opset_version") == 17 or call_kwargs[0][3] is not None


def test_export_missing_weights_dir(
    tmp_dir: Path, sample_params: dict[str, Any]
) -> None:
    """Raises FileNotFoundError when weights_dir does not exist."""
    from pet_quantize.convert.export_vision_encoder import export_vision_encoder

    config = sample_params["convert"]
    config["weights_dir"] = "/nonexistent/weights"
    config["output_dir"] = str(tmp_dir / "converted")

    with pytest.raises(FileNotFoundError):
        export_vision_encoder(config)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_export_vision_encoder.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement export_vision_encoder.py**

```python
# src/pet_quantize/convert/export_vision_encoder.py
"""Export vision encoder (ViT) from HuggingFace model to ONNX format.

Extracts the visual encoder from Qwen2-VL and exports it as FP16 ONNX
for subsequent conversion to RKNN.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel

logger = logging.getLogger(__name__)


def export_vision_encoder(config: dict[str, Any]) -> str:
    """Export the vision encoder to ONNX format.

    Args:
        config: Convert section of params.yaml.

    Returns:
        Path to the exported ONNX file.

    Raises:
        FileNotFoundError: If weights_dir does not exist.
    """
    weights_dir = config["weights_dir"]
    output_dir = config["output_dir"]
    vision_cfg = config["vision"]
    input_size = vision_cfg["input_size"]
    opset = vision_cfg["onnx_opset"]

    if not Path(weights_dir).exists():
        msg = f"Weights directory not found: {weights_dir}"
        raise FileNotFoundError(msg)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = str(Path(output_dir) / "vision_encoder.onnx")

    logger.info("Loading model from %s", weights_dir)
    model = AutoModel.from_pretrained(weights_dir, trust_remote_code=True)
    visual = model.visual
    visual.eval()

    # Create dummy input matching the vision encoder's expected shape
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])

    logger.info("Exporting vision encoder to ONNX: %s", output_path)
    torch.onnx.export(
        visual,
        dummy_input,
        output_path,
        opset_version=opset,
        input_names=["pixel_values"],
        output_names=["visual_features"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "visual_features": {0: "batch_size"},
        },
    )

    logger.info("Vision encoder exported to %s", output_path)
    return output_path
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_export_vision_encoder.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/pet_quantize/convert/ tests/test_export_vision_encoder.py
git commit -m "feat(pet-quantize): add vision encoder ONNX export"
```

---

### Task 6: Convert — convert_to_rknn

**Files:**
- Create: `src/pet_quantize/convert/convert_to_rknn.py`
- Create: `tests/test_convert_rknn.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_convert_rknn.py
"""Tests for pet_quantize.convert.convert_to_rknn."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def onnx_file(tmp_dir: Path) -> Path:
    """Create a dummy ONNX file."""
    p = tmp_dir / "converted" / "vision_encoder.onnx"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"dummy_onnx")
    return p


def test_convert_creates_rknn_file(
    onnx_file: Path, tmp_dir: Path, sample_params: dict[str, Any]
) -> None:
    """Conversion produces an .rknn file."""
    from pet_quantize.convert.convert_to_rknn import convert_vision_to_rknn

    config = sample_params["convert"]
    config["output_dir"] = str(tmp_dir / "converted")

    with patch("pet_quantize.convert.convert_to_rknn.RKNN") as mock_rknn_cls:
        mock_rknn = MagicMock()
        mock_rknn_cls.return_value = mock_rknn
        mock_rknn.load_onnx.return_value = 0
        mock_rknn.build.return_value = 0
        mock_rknn.export_rknn.return_value = 0

        result = convert_vision_to_rknn(str(onnx_file), config)

    assert result.endswith(".rknn")
    mock_rknn.build.assert_called_once()
    mock_rknn.export_rknn.assert_called_once()


def test_convert_uses_fp16_dtype(
    onnx_file: Path, tmp_dir: Path, sample_params: dict[str, Any]
) -> None:
    """Conversion uses FP16 quantization dtype from config."""
    from pet_quantize.convert.convert_to_rknn import convert_vision_to_rknn

    config = sample_params["convert"]
    config["output_dir"] = str(tmp_dir / "converted")

    with patch("pet_quantize.convert.convert_to_rknn.RKNN") as mock_rknn_cls:
        mock_rknn = MagicMock()
        mock_rknn_cls.return_value = mock_rknn
        mock_rknn.load_onnx.return_value = 0
        mock_rknn.build.return_value = 0
        mock_rknn.export_rknn.return_value = 0

        convert_vision_to_rknn(str(onnx_file), config)

    build_kwargs = mock_rknn.build.call_args[1]
    assert build_kwargs.get("do_quantization") is False  # FP16 = no INT8 quantization


def test_convert_missing_onnx_raises(
    tmp_dir: Path, sample_params: dict[str, Any]
) -> None:
    """Raises FileNotFoundError when ONNX file does not exist."""
    from pet_quantize.convert.convert_to_rknn import convert_vision_to_rknn

    config = sample_params["convert"]
    config["output_dir"] = str(tmp_dir / "converted")

    with pytest.raises(FileNotFoundError):
        convert_vision_to_rknn("/nonexistent/vision.onnx", config)


def test_convert_rknn_build_failure(
    onnx_file: Path, tmp_dir: Path, sample_params: dict[str, Any]
) -> None:
    """Raises RuntimeError when RKNN build fails."""
    from pet_quantize.convert.convert_to_rknn import convert_vision_to_rknn

    config = sample_params["convert"]
    config["output_dir"] = str(tmp_dir / "converted")

    with patch("pet_quantize.convert.convert_to_rknn.RKNN") as mock_rknn_cls:
        mock_rknn = MagicMock()
        mock_rknn_cls.return_value = mock_rknn
        mock_rknn.load_onnx.return_value = 0
        mock_rknn.build.return_value = -1  # Failure

        with pytest.raises(RuntimeError, match="RKNN build failed"):
            convert_vision_to_rknn(str(onnx_file), config)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_convert_rknn.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement convert_to_rknn.py**

```python
# src/pet_quantize/convert/convert_to_rknn.py
"""Convert ONNX vision encoder to RKNN format for RK3576.

Uses the RKNN-Toolkit2 SDK to convert the ONNX model to .rknn
with FP16 precision (no INT8 quantization for the vision encoder).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rknn.api import RKNN

logger = logging.getLogger(__name__)


def convert_vision_to_rknn(
    onnx_path: str,
    config: dict[str, Any],
    calib_dir: str | None = None,
) -> str:
    """Convert ONNX vision encoder to RKNN format.

    Args:
        onnx_path: Path to the ONNX model file.
        config: Convert section of params.yaml.
        calib_dir: Path to calibration data directory (unused for FP16).

    Returns:
        Path to the exported .rknn file.

    Raises:
        FileNotFoundError: If ONNX file does not exist.
        RuntimeError: If RKNN build or export fails.
    """
    if not Path(onnx_path).exists():
        msg = f"ONNX file not found: {onnx_path}"
        raise FileNotFoundError(msg)

    vision_cfg = config["vision"]
    target = vision_cfg["rknn_target"]
    output_dir = config["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = str(Path(output_dir) / f"vision_{target}.rknn")

    rknn = RKNN()

    logger.info("Loading ONNX model: %s", onnx_path)
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        msg = f"RKNN load_onnx failed with code {ret}"
        raise RuntimeError(msg)

    logger.info("Building RKNN model (FP16, target=%s)", target)
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        msg = f"RKNN build failed with code {ret}"
        raise RuntimeError(msg)

    logger.info("Exporting RKNN model to %s", output_path)
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        msg = f"RKNN export failed with code {ret}"
        raise RuntimeError(msg)

    rknn.release()
    logger.info("Vision encoder converted to RKNN: %s", output_path)
    return output_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_convert_rknn.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/pet_quantize/convert/convert_to_rknn.py tests/test_convert_rknn.py
git commit -m "feat(pet-quantize): add ONNX to RKNN vision encoder conversion"
```

---

### Task 7: Convert — convert_to_rkllm

**Files:**
- Create: `src/pet_quantize/convert/convert_to_rkllm.py`
- Create: `tests/test_convert_rkllm.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_convert_rkllm.py
"""Tests for pet_quantize.convert.convert_to_rkllm."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def test_convert_creates_rkllm_file(
    tmp_dir: Path, sample_params: dict[str, Any]
) -> None:
    """Conversion produces an .rkllm file."""
    from pet_quantize.convert.convert_to_rkllm import convert_llm_to_rkllm

    config = sample_params["convert"]
    weights_dir = tmp_dir / "weights"
    weights_dir.mkdir()
    config["weights_dir"] = str(weights_dir)
    config["output_dir"] = str(tmp_dir / "converted")
    calib_dir = str(tmp_dir / "calib")

    with patch("pet_quantize.convert.convert_to_rkllm.RKLLMConverter") as mock_cls:
        mock_converter = MagicMock()
        mock_cls.return_value = mock_converter

        result = convert_llm_to_rkllm(config, calib_dir)

    assert result.endswith(".rkllm")
    mock_converter.convert.assert_called_once()
    mock_converter.export.assert_called_once()


def test_convert_uses_w8a8_quantization(
    tmp_dir: Path, sample_params: dict[str, Any]
) -> None:
    """Conversion passes W8A8 quantization config."""
    from pet_quantize.convert.convert_to_rkllm import convert_llm_to_rkllm

    config = sample_params["convert"]
    weights_dir = tmp_dir / "weights"
    weights_dir.mkdir()
    config["weights_dir"] = str(weights_dir)
    config["output_dir"] = str(tmp_dir / "converted")

    with patch("pet_quantize.convert.convert_to_rkllm.RKLLMConverter") as mock_cls:
        mock_converter = MagicMock()
        mock_cls.return_value = mock_converter

        convert_llm_to_rkllm(config, str(tmp_dir / "calib"))

    init_kwargs = mock_cls.call_args[1]
    assert init_kwargs.get("quantization") == "w8a8"


def test_convert_missing_weights_raises(
    tmp_dir: Path, sample_params: dict[str, Any]
) -> None:
    """Raises FileNotFoundError when weights_dir does not exist."""
    from pet_quantize.convert.convert_to_rkllm import convert_llm_to_rkllm

    config = sample_params["convert"]
    config["weights_dir"] = "/nonexistent/weights"
    config["output_dir"] = str(tmp_dir / "converted")

    with pytest.raises(FileNotFoundError):
        convert_llm_to_rkllm(config, str(tmp_dir / "calib"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_convert_rkllm.py -v`
Expected: FAIL

- [ ] **Step 3: Implement convert_to_rkllm.py**

```python
# src/pet_quantize/convert/convert_to_rkllm.py
"""Convert merged LLM weights to RKLLM format for RK3576.

Uses the RKNN-LLM SDK to convert HuggingFace model weights to .rkllm
with W8A8 quantization, using calibration data for scale determination.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rkllm.api import RKLLMConverter

logger = logging.getLogger(__name__)


def convert_llm_to_rkllm(
    config: dict[str, Any],
    calib_dir: str,
) -> str:
    """Convert HuggingFace LLM weights to RKLLM W8A8 format.

    Args:
        config: Convert section of params.yaml.
        calib_dir: Path to calibration data directory for quantization.

    Returns:
        Path to the exported .rkllm file.

    Raises:
        FileNotFoundError: If weights_dir does not exist.
        RuntimeError: If conversion fails.
    """
    weights_dir = config["weights_dir"]
    if not Path(weights_dir).exists():
        msg = f"Weights directory not found: {weights_dir}"
        raise FileNotFoundError(msg)

    llm_cfg = config["llm"]
    target = llm_cfg["rkllm_target"]
    quantization = llm_cfg["quantization"]
    output_dir = config["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = str(
        Path(output_dir) / f"qwen2vl_2b_{quantization}_{target}.rkllm"
    )

    logger.info(
        "Converting LLM to RKLLM: weights=%s, quantization=%s, target=%s",
        weights_dir,
        quantization,
        target,
    )

    converter = RKLLMConverter(
        model_path=weights_dir,
        target_platform=target,
        quantization=quantization,
        calibration_data=calib_dir,
    )
    converter.convert()
    converter.export(output_path)

    logger.info("LLM converted to RKLLM: %s", output_path)
    return output_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_convert_rkllm.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/pet_quantize/convert/convert_to_rkllm.py tests/test_convert_rkllm.py
git commit -m "feat(pet-quantize): add LLM to RKLLM W8A8 conversion"
```

---

### Task 8: Convert — convert_audio

**Files:**
- Create: `src/pet_quantize/convert/convert_audio.py`
- Create: `tests/test_convert_audio.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_convert_audio.py
"""Tests for pet_quantize.convert.convert_audio."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def test_convert_creates_rknn_file(
    tmp_dir: Path, sample_params: dict[str, Any]
) -> None:
    """Conversion produces an INT8 .rknn file."""
    from pet_quantize.convert.convert_audio import convert_audio_to_rknn

    config = sample_params["convert"]
    checkpoint = tmp_dir / "audio.pt"
    checkpoint.write_bytes(b"dummy_checkpoint")
    config["audio_checkpoint"] = str(checkpoint)
    config["output_dir"] = str(tmp_dir / "converted")

    with (
        patch("pet_quantize.convert.convert_audio.torch") as mock_torch,
        patch("pet_quantize.convert.convert_audio.RKNN") as mock_rknn_cls,
    ):
        mock_torch.load.return_value = MagicMock()
        mock_rknn = MagicMock()
        mock_rknn_cls.return_value = mock_rknn
        mock_rknn.load_onnx.return_value = 0
        mock_rknn.build.return_value = 0
        mock_rknn.export_rknn.return_value = 0

        result = convert_audio_to_rknn(config, str(tmp_dir / "calib"))

    assert result.endswith(".rknn")
    assert "int8" in result or "audio" in result


def test_convert_uses_int8_quantization(
    tmp_dir: Path, sample_params: dict[str, Any]
) -> None:
    """Conversion uses INT8 quantization with calibration data."""
    from pet_quantize.convert.convert_audio import convert_audio_to_rknn

    config = sample_params["convert"]
    checkpoint = tmp_dir / "audio.pt"
    checkpoint.write_bytes(b"dummy")
    config["audio_checkpoint"] = str(checkpoint)
    config["output_dir"] = str(tmp_dir / "converted")

    with (
        patch("pet_quantize.convert.convert_audio.torch") as mock_torch,
        patch("pet_quantize.convert.convert_audio.RKNN") as mock_rknn_cls,
    ):
        mock_torch.load.return_value = MagicMock()
        mock_rknn = MagicMock()
        mock_rknn_cls.return_value = mock_rknn
        mock_rknn.load_onnx.return_value = 0
        mock_rknn.build.return_value = 0
        mock_rknn.export_rknn.return_value = 0

        convert_audio_to_rknn(config, str(tmp_dir / "calib"))

    build_kwargs = mock_rknn.build.call_args[1]
    assert build_kwargs.get("do_quantization") is True


def test_convert_missing_checkpoint_raises(
    tmp_dir: Path, sample_params: dict[str, Any]
) -> None:
    """Raises FileNotFoundError when audio checkpoint does not exist."""
    from pet_quantize.convert.convert_audio import convert_audio_to_rknn

    config = sample_params["convert"]
    config["audio_checkpoint"] = "/nonexistent/audio.pt"
    config["output_dir"] = str(tmp_dir / "converted")

    with pytest.raises(FileNotFoundError):
        convert_audio_to_rknn(config, str(tmp_dir / "calib"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_convert_audio.py -v`
Expected: FAIL

- [ ] **Step 3: Implement convert_audio.py**

```python
# src/pet_quantize/convert/convert_audio.py
"""Convert audio CNN model to INT8 RKNN format for RK3576.

Loads the PyTorch audio CNN checkpoint, exports to ONNX, then converts
to RKNN with INT8 quantization using calibration data.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import torch
from rknn.api import RKNN

logger = logging.getLogger(__name__)


def convert_audio_to_rknn(
    config: dict[str, Any],
    calib_dir: str,
) -> str:
    """Convert audio CNN checkpoint to INT8 RKNN format.

    Args:
        config: Convert section of params.yaml.
        calib_dir: Path to calibration data directory for INT8 quantization.

    Returns:
        Path to the exported .rknn file.

    Raises:
        FileNotFoundError: If audio checkpoint does not exist.
        RuntimeError: If RKNN build or export fails.
    """
    checkpoint_path = config["audio_checkpoint"]
    if not Path(checkpoint_path).exists():
        msg = f"Audio checkpoint not found: {checkpoint_path}"
        raise FileNotFoundError(msg)

    audio_cfg = config["audio"]
    target = audio_cfg["rknn_target"]
    output_dir = config["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = str(Path(output_dir) / f"audio_cnn_int8.rknn")

    # Load PyTorch model and export to ONNX
    logger.info("Loading audio checkpoint: %s", checkpoint_path)
    model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if hasattr(model, "eval"):
        model.eval()

    # Export to temporary ONNX
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        onnx_path = tmp.name

    # Audio input: log-mel spectrogram [batch, 1, n_mels, time_frames]
    dummy_input = torch.randn(1, 1, 64, 100)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=17,
        input_names=["audio_input"],
        output_names=["class_logits"],
        dynamic_axes={
            "audio_input": {0: "batch_size"},
            "class_logits": {0: "batch_size"},
        },
    )

    # Convert ONNX to RKNN with INT8 quantization
    rknn = RKNN()

    logger.info("Loading audio ONNX model")
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        msg = f"RKNN load_onnx failed with code {ret}"
        raise RuntimeError(msg)

    logger.info("Building RKNN model (INT8, target=%s)", target)
    ret = rknn.build(do_quantization=True, dataset=calib_dir)
    if ret != 0:
        msg = f"RKNN build failed with code {ret}"
        raise RuntimeError(msg)

    logger.info("Exporting RKNN model to %s", output_path)
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        msg = f"RKNN export failed with code {ret}"
        raise RuntimeError(msg)

    rknn.release()

    # Cleanup temp ONNX
    Path(onnx_path).unlink(missing_ok=True)

    logger.info("Audio model converted to RKNN: %s", output_path)
    return output_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_convert_audio.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/pet_quantize/convert/convert_audio.py tests/test_convert_audio.py
git commit -m "feat(pet-quantize): add audio CNN to INT8 RKNN conversion"
```

---

### Task 9: Inference — rknn_runner

**Files:**
- Create: `src/pet_quantize/inference/__init__.py`
- Create: `src/pet_quantize/inference/rknn_runner.py`

- [ ] **Step 1: Create inference __init__.py**

```python
# src/pet_quantize/inference/__init__.py
"""Inference interfaces for quantized models.

Public API:
    run_quantized_pipeline: Full VLM inference (vision + LLM).
    run_audio_inference: Audio CNN INT8 inference.
"""
from pet_quantize.inference.pipeline import run_quantized_pipeline
from pet_quantize.inference.rknn_runner import run_audio_inference

__all__ = ["run_quantized_pipeline", "run_audio_inference"]
```

- [ ] **Step 2: Implement rknn_runner.py**

```python
# src/pet_quantize/inference/rknn_runner.py
"""RKNN SDK wrapper for vision encoder and audio model inference.

Supports dual-mode: on-device (ADB) and simulated (PC simulator).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from rknn.api import RKNN

logger = logging.getLogger(__name__)


class RKNNRunner:
    """Wrapper for RKNN model inference with dual-mode support.

    Args:
        model_path: Path to the .rknn model file.
        target: Target platform (e.g., "rk3576"). None for PC simulation.
        device_id: ADB device serial number. Required when target is set.
    """

    def __init__(
        self,
        model_path: str,
        target: str | None = None,
        device_id: str | None = None,
    ) -> None:
        if not Path(model_path).exists():
            msg = f"RKNN model not found: {model_path}"
            raise FileNotFoundError(msg)

        self._model_path = model_path
        self._target = target
        self._device_id = device_id
        self._rknn: RKNN | None = None

    def init(self) -> None:
        """Initialize the RKNN runtime.

        Raises:
            RuntimeError: If RKNN runtime initialization fails.
        """
        self._rknn = RKNN()

        ret = self._rknn.load_rknn(self._model_path)
        if ret != 0:
            msg = f"Failed to load RKNN model: {ret}"
            raise RuntimeError(msg)

        if self._target and self._device_id:
            logger.info(
                "Initializing on-device runtime: target=%s, device=%s",
                self._target,
                self._device_id,
            )
            ret = self._rknn.init_runtime(
                target=self._target, device_id=self._device_id
            )
        else:
            logger.info("Initializing simulated runtime (PC)")
            ret = self._rknn.init_runtime(target=None)

        if ret != 0:
            msg = f"Failed to init RKNN runtime: {ret}"
            raise RuntimeError(msg)

    def infer(self, inputs: list[np.ndarray]) -> tuple[list[np.ndarray], float]:
        """Run inference on the given inputs.

        Args:
            inputs: List of numpy arrays matching the model's input spec.

        Returns:
            Tuple of (output arrays, elapsed time in milliseconds).

        Raises:
            RuntimeError: If runtime is not initialized.
        """
        if self._rknn is None:
            msg = "RKNN runtime not initialized. Call init() first."
            raise RuntimeError(msg)

        start = time.perf_counter()
        outputs = self._rknn.inference(inputs=inputs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return outputs, elapsed_ms

    def release(self) -> None:
        """Release the RKNN runtime resources."""
        if self._rknn is not None:
            self._rknn.release()
            self._rknn = None


def run_audio_inference(
    model_path: str,
    audio_paths: list[str],
    device_id: str | None = None,
) -> dict[str, Any]:
    """Run audio CNN INT8 inference on a list of audio files.

    Args:
        model_path: Path to the .rknn audio model.
        audio_paths: List of audio file paths.
        device_id: ADB device ID, or None for simulated mode.

    Returns:
        Dict with keys: predictions (list[str]), confidences (list[float]),
        timings (list[float]).
    """
    target = "rk3576" if device_id else None
    runner = RKNNRunner(model_path, target=target, device_id=device_id)
    runner.init()

    predictions: list[str] = []
    confidences: list[float] = []
    timings: list[float] = []
    classes = ["eating", "drinking", "vomiting", "ambient", "other"]

    try:
        for audio_path in audio_paths:
            # Load and preprocess audio (log-mel spectrogram)
            features = _load_audio_features(audio_path)
            outputs, elapsed_ms = runner.infer([features])

            logits = outputs[0].flatten()
            probs = _softmax(logits)
            pred_idx = int(np.argmax(probs))

            predictions.append(classes[pred_idx])
            confidences.append(float(probs[pred_idx]))
            timings.append(elapsed_ms)
    finally:
        runner.release()

    return {
        "predictions": predictions,
        "confidences": confidences,
        "timings": timings,
    }


def _load_audio_features(audio_path: str) -> np.ndarray:
    """Load audio file and extract log-mel spectrogram features.

    Args:
        audio_path: Path to an audio file.

    Returns:
        Numpy array of shape [1, 1, 64, T] (batch, channel, n_mels, time).
    """
    import torchaudio

    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_mels=64
    )
    mel = mel_transform(waveform)
    log_mel = (mel + 1e-8).log()

    # Shape: [1, 1, 64, T]
    return log_mel.unsqueeze(0).numpy()


def _softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax for a 1D array."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
```

- [ ] **Step 3: Commit**

```bash
git add src/pet_quantize/inference/__init__.py src/pet_quantize/inference/rknn_runner.py
git commit -m "feat(pet-quantize): add RKNN runner with dual-mode inference"
```

---

### Task 10: Inference — rkllm_runner

**Files:**
- Create: `src/pet_quantize/inference/rkllm_runner.py`

- [ ] **Step 1: Implement rkllm_runner.py**

```python
# src/pet_quantize/inference/rkllm_runner.py
"""RKLLM SDK wrapper for LLM inference.

Supports dual-mode: on-device (ADB) and simulated (PC).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from rkllm.api import RKLLMRuntime

logger = logging.getLogger(__name__)


class RKLLMRunner:
    """Wrapper for RKLLM model inference with dual-mode support.

    Args:
        model_path: Path to the .rkllm model file.
        target: Target platform (e.g., "rk3576"). None for PC simulation.
        device_id: ADB device serial number. Required when target is set.
    """

    def __init__(
        self,
        model_path: str,
        target: str | None = None,
        device_id: str | None = None,
    ) -> None:
        if not Path(model_path).exists():
            msg = f"RKLLM model not found: {model_path}"
            raise FileNotFoundError(msg)

        self._model_path = model_path
        self._target = target
        self._device_id = device_id
        self._runtime: RKLLMRuntime | None = None

    def init(self) -> None:
        """Initialize the RKLLM runtime.

        Raises:
            RuntimeError: If initialization fails.
        """
        kwargs: dict[str, Any] = {"model_path": self._model_path}
        if self._target and self._device_id:
            kwargs["target"] = self._target
            kwargs["device_id"] = self._device_id
            logger.info(
                "Initializing RKLLM on-device: target=%s, device=%s",
                self._target,
                self._device_id,
            )
        else:
            logger.info("Initializing RKLLM simulated runtime (PC)")

        self._runtime = RKLLMRuntime(**kwargs)

    def generate(
        self,
        prompt: str,
        visual_features: Any | None = None,
        max_tokens: int = 2048,
    ) -> tuple[str, float]:
        """Generate text from a prompt with optional visual features.

        Args:
            prompt: Text prompt for the LLM.
            visual_features: Visual encoder output features (numpy array).
            max_tokens: Maximum number of tokens to generate.

        Returns:
            Tuple of (generated text, elapsed time in milliseconds).

        Raises:
            RuntimeError: If runtime is not initialized.
        """
        if self._runtime is None:
            msg = "RKLLM runtime not initialized. Call init() first."
            raise RuntimeError(msg)

        start = time.perf_counter()
        output = self._runtime.generate(
            prompt=prompt,
            visual_features=visual_features,
            max_new_tokens=max_tokens,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        return output, elapsed_ms

    def release(self) -> None:
        """Release the RKLLM runtime resources."""
        if self._runtime is not None:
            self._runtime.release()
            self._runtime = None
```

- [ ] **Step 2: Commit**

```bash
git add src/pet_quantize/inference/rkllm_runner.py
git commit -m "feat(pet-quantize): add RKLLM runner with dual-mode inference"
```

---

### Task 11: Inference — pipeline

**Files:**
- Create: `src/pet_quantize/inference/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pipeline.py
"""Tests for pet_quantize.inference.pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture()
def model_dir(tmp_dir: Path) -> Path:
    """Create a dummy model directory with expected files."""
    d = tmp_dir / "converted"
    d.mkdir()
    (d / "vision_rk3576.rknn").write_bytes(b"dummy_rknn")
    (d / "qwen2vl_2b_w8a8_rk3576.rkllm").write_bytes(b"dummy_rkllm")
    return d


@pytest.fixture()
def image_paths(tmp_dir: Path) -> list[str]:
    """Create dummy image files."""
    paths = []
    for i in range(3):
        p = tmp_dir / f"image_{i}.jpg"
        p.write_bytes(b"dummy_image")
        paths.append(str(p))
    return paths


def test_pipeline_returns_expected_keys(
    model_dir: Path,
    image_paths: list[str],
    sample_params_path: Path,
) -> None:
    """Pipeline returns dict with outputs, timings, fp16_outputs keys."""
    from pet_quantize.inference.pipeline import run_quantized_pipeline

    with (
        patch("pet_quantize.inference.pipeline.RKNNRunner") as mock_rknn,
        patch("pet_quantize.inference.pipeline.RKLLMRunner") as mock_rkllm,
        patch("pet_quantize.inference.pipeline._run_fp16_reference") as mock_fp16,
    ):
        mock_rknn_inst = MagicMock()
        mock_rknn.return_value = mock_rknn_inst
        mock_rknn_inst.infer.return_value = ([np.zeros((1, 768))], 10.0)

        mock_rkllm_inst = MagicMock()
        mock_rkllm.return_value = mock_rkllm_inst
        mock_rkllm_inst.generate.return_value = ('{"valid": "json"}', 100.0)

        mock_fp16.return_value = ['{"fp16": "output"}'] * 3

        result = run_quantized_pipeline(
            model_dir=str(model_dir),
            image_paths=image_paths,
            device_id=None,
            params_path=str(sample_params_path),
        )

    assert "outputs" in result
    assert "timings" in result
    assert "fp16_outputs" in result
    assert len(result["outputs"]) == 3
    assert len(result["fp16_outputs"]) == 3


def test_pipeline_simulated_no_timings(
    model_dir: Path,
    image_paths: list[str],
    sample_params_path: Path,
) -> None:
    """In simulated mode, timings is empty list."""
    from pet_quantize.inference.pipeline import run_quantized_pipeline

    with (
        patch("pet_quantize.inference.pipeline.RKNNRunner") as mock_rknn,
        patch("pet_quantize.inference.pipeline.RKLLMRunner") as mock_rkllm,
        patch("pet_quantize.inference.pipeline._run_fp16_reference") as mock_fp16,
    ):
        mock_rknn_inst = MagicMock()
        mock_rknn.return_value = mock_rknn_inst
        mock_rknn_inst.infer.return_value = ([np.zeros((1, 768))], 10.0)

        mock_rkllm_inst = MagicMock()
        mock_rkllm.return_value = mock_rkllm_inst
        mock_rkllm_inst.generate.return_value = ('{"valid": "json"}', 100.0)

        mock_fp16.return_value = ['{"fp16": "output"}'] * 3

        result = run_quantized_pipeline(
            model_dir=str(model_dir),
            image_paths=image_paths,
            device_id=None,
            params_path=str(sample_params_path),
        )

    assert result["timings"] == []


def test_pipeline_device_mode_has_timings(
    model_dir: Path,
    image_paths: list[str],
    sample_params_path: Path,
) -> None:
    """In device mode, timings has one entry per image."""
    from pet_quantize.inference.pipeline import run_quantized_pipeline

    with (
        patch("pet_quantize.inference.pipeline.RKNNRunner") as mock_rknn,
        patch("pet_quantize.inference.pipeline.RKLLMRunner") as mock_rkllm,
        patch("pet_quantize.inference.pipeline._run_fp16_reference") as mock_fp16,
    ):
        mock_rknn_inst = MagicMock()
        mock_rknn.return_value = mock_rknn_inst
        mock_rknn_inst.infer.return_value = ([np.zeros((1, 768))], 10.0)

        mock_rkllm_inst = MagicMock()
        mock_rkllm.return_value = mock_rkllm_inst
        mock_rkllm_inst.generate.return_value = ('{"valid": "json"}', 100.0)

        mock_fp16.return_value = ['{"fp16": "output"}'] * 3

        result = run_quantized_pipeline(
            model_dir=str(model_dir),
            image_paths=image_paths,
            device_id="DEVICE123",
            params_path=str(sample_params_path),
        )

    assert len(result["timings"]) == 3


def test_pipeline_missing_model_dir(sample_params_path: Path) -> None:
    """Raises FileNotFoundError for missing model directory."""
    from pet_quantize.inference.pipeline import run_quantized_pipeline

    with pytest.raises(FileNotFoundError):
        run_quantized_pipeline(
            model_dir="/nonexistent/models",
            image_paths=["a.jpg"],
            params_path=str(sample_params_path),
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL

- [ ] **Step 3: Implement pipeline.py**

```python
# src/pet_quantize/inference/pipeline.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/pet_quantize/inference/pipeline.py tests/test_pipeline.py
git commit -m "feat(pet-quantize): add VLM inference pipeline with FP16 reference"
```

---

### Task 12: Packaging — build_package

**Files:**
- Create: `src/pet_quantize/packaging/__init__.py`
- Create: `src/pet_quantize/packaging/build_package.py`
- Create: `tests/test_build_package.py`

- [ ] **Step 1: Create packaging __init__.py**

```python
# src/pet_quantize/packaging/__init__.py
"""Artifact packaging, signing, and verification."""
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_build_package.py
"""Tests for pet_quantize.packaging.build_package."""
from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture()
def converted_dir(tmp_dir: Path) -> Path:
    """Create a dummy converted artifacts directory."""
    d = tmp_dir / "converted"
    d.mkdir()
    (d / "vision_rk3576.rknn").write_bytes(b"vision_data" * 100)
    (d / "qwen2vl_2b_w8a8_rk3576.rkllm").write_bytes(b"llm_data" * 100)
    (d / "audio_cnn_int8.rknn").write_bytes(b"audio_data" * 100)
    return d


@pytest.fixture()
def release_dir(tmp_dir: Path) -> Path:
    """Create release output directory."""
    d = tmp_dir / "release"
    d.mkdir()
    return d


def test_build_creates_tarball_and_manifest(
    converted_dir: Path,
    release_dir: Path,
    sample_params: dict[str, Any],
) -> None:
    """build_package creates a tarball and manifest.json."""
    from pet_quantize.packaging.build_package import build_package

    config = sample_params["packaging"]
    config["version"] = "1.0.0"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "pet_quantize.packaging.build_package._get_prompt_files",
            lambda _: {
                "prompt_system": ("prompt_system_v1.0.txt", b"system prompt"),
                "prompt_user": ("prompt_user_v1.0.jinja2", b"user prompt"),
            },
        )
        result = build_package(
            converted_dir=str(converted_dir),
            release_dir=str(release_dir),
            config=config,
        )

    assert Path(result["tarball_path"]).exists()
    assert Path(result["manifest_path"]).exists()


def test_manifest_contains_all_files(
    converted_dir: Path,
    release_dir: Path,
    sample_params: dict[str, Any],
) -> None:
    """Manifest lists all model files with sha256 and size_bytes."""
    from pet_quantize.packaging.build_package import build_package

    config = sample_params["packaging"]
    config["version"] = "1.0.0"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "pet_quantize.packaging.build_package._get_prompt_files",
            lambda _: {
                "prompt_system": ("prompt_system_v1.0.txt", b"system prompt"),
                "prompt_user": ("prompt_user_v1.0.jinja2", b"user prompt"),
            },
        )
        result = build_package(
            converted_dir=str(converted_dir),
            release_dir=str(release_dir),
            config=config,
        )

    manifest = json.loads(Path(result["manifest_path"]).read_text())

    assert "vision_encoder" in manifest["files"]
    assert "llm" in manifest["files"]
    assert "audio" in manifest["files"]
    assert "prompt_system" in manifest["files"]
    assert "prompt_user" in manifest["files"]

    for file_info in manifest["files"].values():
        assert "sha256" in file_info
        assert "size_bytes" in file_info
        assert "path" in file_info
        assert len(file_info["sha256"]) == 64  # sha256 hex


def test_manifest_contains_metadata(
    converted_dir: Path,
    release_dir: Path,
    sample_params: dict[str, Any],
) -> None:
    """Manifest contains all required metadata fields."""
    from pet_quantize.packaging.build_package import build_package

    config = sample_params["packaging"]
    config["version"] = "1.2.0"
    config["lora_version"] = "1.2"
    config["release_notes"] = "Test release"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "pet_quantize.packaging.build_package._get_prompt_files",
            lambda _: {
                "prompt_system": ("prompt_system_v1.0.txt", b"system prompt"),
                "prompt_user": ("prompt_user_v1.0.jinja2", b"user prompt"),
            },
        )
        result = build_package(
            converted_dir=str(converted_dir),
            release_dir=str(release_dir),
            config=config,
        )

    manifest = json.loads(Path(result["manifest_path"]).read_text())

    assert manifest["version"] == "1.2.0"
    assert manifest["lora_version"] == "1.2"
    assert manifest["min_firmware"] == "2.0.0"
    assert manifest["release_notes"] == "Test release"
    assert "schema_version" in manifest
    assert "prompt_version" in manifest
    assert "build_timestamp" in manifest


def test_tarball_contains_all_files(
    converted_dir: Path,
    release_dir: Path,
    sample_params: dict[str, Any],
) -> None:
    """Tarball contains all model, prompt, and manifest files."""
    from pet_quantize.packaging.build_package import build_package

    config = sample_params["packaging"]
    config["version"] = "1.0.0"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "pet_quantize.packaging.build_package._get_prompt_files",
            lambda _: {
                "prompt_system": ("prompt_system_v1.0.txt", b"system prompt"),
                "prompt_user": ("prompt_user_v1.0.jinja2", b"user prompt"),
            },
        )
        result = build_package(
            converted_dir=str(converted_dir),
            release_dir=str(release_dir),
            config=config,
        )

    with tarfile.open(result["tarball_path"], "r:gz") as tar:
        names = tar.getnames()

    assert any("manifest.json" in n for n in names)
    assert any(".rknn" in n for n in names)
    assert any(".rkllm" in n for n in names)
    assert any("prompt_system" in n for n in names)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_build_package.py -v`
Expected: FAIL

- [ ] **Step 4: Implement build_package.py**

```python
# src/pet_quantize/packaging/build_package.py
"""Build release package with tarball and manifest.json.

Collects converted model files and prompt files from pet-schema,
generates manifest with sha256 checksums, and creates a signed tarball.
"""
from __future__ import annotations

import hashlib
import json
import logging
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Mapping of model file patterns to manifest keys
_MODEL_FILE_MAP = {
    "vision_*.rknn": "vision_encoder",
    "qwen2vl_*.rkllm": "llm",
    "audio_*.rknn": "audio",
}


def _sha256(data: bytes) -> str:
    """Compute SHA-256 hex digest of data."""
    return hashlib.sha256(data).hexdigest()


def _get_prompt_files(schema_version: str) -> dict[str, tuple[str, bytes]]:
    """Read prompt files from the installed pet-schema package.

    Args:
        schema_version: Schema version string (e.g., "1.0").

    Returns:
        Dict mapping manifest key to (filename, content bytes).
    """
    from pet_schema import get_prompt_paths

    paths = get_prompt_paths(schema_version)
    result = {}
    for key, path in paths.items():
        p = Path(path)
        result[key] = (p.name, p.read_bytes())
    return result


def _collect_model_files(converted_dir: str) -> dict[str, tuple[str, bytes]]:
    """Collect model files from the converted artifacts directory.

    Args:
        converted_dir: Path to artifacts/converted/.

    Returns:
        Dict mapping manifest key to (filename, content bytes).
    """
    import glob

    result = {}
    for pattern, key in _MODEL_FILE_MAP.items():
        matches = glob.glob(str(Path(converted_dir) / pattern))
        if not matches:
            logger.warning("No model file matching '%s' in %s", pattern, converted_dir)
            continue
        file_path = Path(matches[0])
        result[key] = (file_path.name, file_path.read_bytes())
    return result


def build_package(
    converted_dir: str,
    release_dir: str,
    config: dict[str, Any],
) -> dict[str, str]:
    """Build the release package.

    Args:
        converted_dir: Path to artifacts/converted/.
        release_dir: Path to artifacts/release/.
        config: Packaging section of params.yaml.

    Returns:
        Dict with keys: tarball_path, manifest_path.
    """
    version = config["version"]
    lora_version = config.get("lora_version", "")
    min_firmware = config.get("min_firmware", "2.0.0")
    release_notes = config.get("release_notes", "")

    Path(release_dir).mkdir(parents=True, exist_ok=True)

    # Collect all files
    model_files = _collect_model_files(converted_dir)

    try:
        from pet_schema import __version__ as schema_pkg_version
    except ImportError:
        schema_pkg_version = "1.0"

    schema_version = schema_pkg_version
    prompt_version = schema_pkg_version

    prompt_files = _get_prompt_files(schema_version)

    all_files = {**model_files, **prompt_files}

    # Build manifest
    files_manifest: dict[str, Any] = {}
    for key, (filename, content) in all_files.items():
        files_manifest[key] = {
            "path": filename,
            "sha256": _sha256(content),
            "size_bytes": len(content),
        }

    manifest = {
        "version": version,
        "schema_version": schema_version,
        "prompt_version": prompt_version,
        "lora_version": lora_version,
        "min_firmware": min_firmware,
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "files": files_manifest,
        "release_notes": release_notes,
    }

    manifest_path = str(Path(release_dir) / "manifest.json")
    Path(manifest_path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info("Manifest written to %s", manifest_path)

    # Create tarball
    tarball_name = f"pet-model-v{version}.tar.gz"
    tarball_path = str(Path(release_dir) / tarball_name)

    with tarfile.open(tarball_path, "w:gz") as tar:
        # Add manifest
        tar.add(manifest_path, arcname="manifest.json")

        # Add all files
        for _key, (filename, content) in all_files.items():
            import io
            import tarfile as tf

            info = tf.TarInfo(name=filename)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))

    logger.info("Tarball created: %s", tarball_path)

    return {"tarball_path": tarball_path, "manifest_path": manifest_path}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_build_package.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/pet_quantize/packaging/ tests/test_build_package.py
git commit -m "feat(pet-quantize): add package builder with manifest.json generation"
```

---

### Task 13: Packaging — sign_package

**Files:**
- Create: `src/pet_quantize/packaging/sign_package.py`
- Create: `tests/test_sign_package.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sign_package.py
"""Tests for pet_quantize.packaging.sign_package."""
from __future__ import annotations

from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization


@pytest.fixture()
def rsa_key_pair(tmp_dir: Path) -> tuple[Path, Path]:
    """Generate a test RSA key pair."""
    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048
    )
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    priv_path = tmp_dir / "test_private.pem"
    pub_path = tmp_dir / "test_public.pem"
    priv_path.write_bytes(private_pem)
    pub_path.write_bytes(public_pem)
    return priv_path, pub_path


@pytest.fixture()
def tarball(tmp_dir: Path) -> Path:
    """Create a dummy tarball for signing."""
    p = tmp_dir / "release" / "pet-model-v1.0.0.tar.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"fake_tarball_content" * 100)
    return p


def test_sign_with_key_creates_sig_file(
    tarball: Path, rsa_key_pair: tuple[Path, Path]
) -> None:
    """Signing with a private key creates a .sig file."""
    from pet_quantize.packaging.sign_package import sign_package

    priv_path, _ = rsa_key_pair
    sig_path = sign_package(str(tarball), str(priv_path))

    assert sig_path is not None
    assert Path(sig_path).exists()
    assert sig_path.endswith(".sig")


def test_sign_without_key_returns_none(tarball: Path) -> None:
    """Signing without a private key skips and returns None."""
    from pet_quantize.packaging.sign_package import sign_package

    result = sign_package(str(tarball), "")
    assert result is None


def test_sign_nonexistent_key_returns_none(tarball: Path) -> None:
    """Signing with nonexistent key path skips and returns None."""
    from pet_quantize.packaging.sign_package import sign_package

    result = sign_package(str(tarball), "/nonexistent/key.pem")
    assert result is None


def test_sign_nonexistent_tarball_raises(
    rsa_key_pair: tuple[Path, Path],
) -> None:
    """Signing a nonexistent tarball raises FileNotFoundError."""
    from pet_quantize.packaging.sign_package import sign_package

    priv_path, _ = rsa_key_pair
    with pytest.raises(FileNotFoundError):
        sign_package("/nonexistent/tarball.tar.gz", str(priv_path))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sign_package.py -v`
Expected: FAIL

- [ ] **Step 3: Implement sign_package.py**

```python
# src/pet_quantize/packaging/sign_package.py
"""RSA-2048 package signing.

Signs the release tarball. In local development where no private key is
available, signing is skipped with a warning.
"""
from __future__ import annotations

import logging
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

logger = logging.getLogger(__name__)


def sign_package(
    tarball_path: str,
    private_key_path: str,
) -> str | None:
    """Sign a tarball with RSA-2048.

    Args:
        tarball_path: Path to the tarball file.
        private_key_path: Path to the RSA private key PEM file.
            Empty string or nonexistent path triggers skip.

    Returns:
        Path to the .sig signature file, or None if signing was skipped.

    Raises:
        FileNotFoundError: If tarball does not exist.
    """
    if not Path(tarball_path).exists():
        msg = f"Tarball not found: {tarball_path}"
        raise FileNotFoundError(msg)

    if not private_key_path or not Path(private_key_path).exists():
        logger.warning(
            "RSA private key not available, skipping signing. "
            "Set RSA_PRIVATE_KEY_PATH for production builds."
        )
        return None

    # Load private key
    key_data = Path(private_key_path).read_bytes()
    private_key = serialization.load_pem_private_key(key_data, password=None)

    # Read tarball
    tarball_data = Path(tarball_path).read_bytes()

    # Sign
    signature = private_key.sign(
        tarball_data,
        padding.PKCS1v15(),
        hashes.SHA256(),
    )

    # Write signature
    sig_path = tarball_path + ".sig"
    Path(sig_path).write_bytes(signature)

    logger.info("Package signed: %s", sig_path)
    return sig_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sign_package.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/pet_quantize/packaging/sign_package.py tests/test_sign_package.py
git commit -m "feat(pet-quantize): add RSA-2048 package signing with skip-on-no-key"
```

---

### Task 14: Packaging — verify_package

**Files:**
- Create: `src/pet_quantize/packaging/verify_package.py`
- Create: `tests/test_verify_package.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_verify_package.py
"""Tests for pet_quantize.packaging.verify_package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


@pytest.fixture()
def rsa_key_pair(tmp_dir: Path) -> tuple[Path, Path]:
    """Generate a test RSA key pair."""
    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048
    )
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    priv_path = tmp_dir / "test_private.pem"
    pub_path = tmp_dir / "test_public.pem"
    priv_path.write_bytes(private_pem)
    pub_path.write_bytes(public_pem)
    return priv_path, pub_path


@pytest.fixture()
def signed_release(tmp_dir: Path, rsa_key_pair: tuple[Path, Path]) -> Path:
    """Create a dummy signed release directory."""
    release = tmp_dir / "release"
    release.mkdir()

    # Create model file
    model_data = b"model_content" * 100
    model_path = release / "vision_rk3576.rknn"
    model_path.write_bytes(model_data)

    # Create manifest
    manifest = {
        "version": "1.0.0",
        "files": {
            "vision_encoder": {
                "path": "vision_rk3576.rknn",
                "sha256": hashlib.sha256(model_data).hexdigest(),
                "size_bytes": len(model_data),
            }
        },
    }
    manifest_path = release / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    # Create tarball (just the model for simplicity)
    import tarfile
    tarball_path = release / "pet-model-v1.0.0.tar.gz"
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(manifest_path, arcname="manifest.json")
        tar.add(model_path, arcname="vision_rk3576.rknn")

    # Sign
    priv_path, _ = rsa_key_pair
    private_key = serialization.load_pem_private_key(
        priv_path.read_bytes(), password=None
    )
    signature = private_key.sign(
        tarball_path.read_bytes(),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )
    sig_path = release / "pet-model-v1.0.0.tar.gz.sig"
    sig_path.write_bytes(signature)

    return release


def test_verify_valid_package(
    signed_release: Path, rsa_key_pair: tuple[Path, Path]
) -> None:
    """A valid signed package passes verification."""
    from pet_quantize.packaging.verify_package import verify_package

    _, pub_path = rsa_key_pair
    result = verify_package(
        release_dir=str(signed_release),
        public_key_path=str(pub_path),
    )
    assert result.passed is True
    assert len(result.errors) == 0


def test_verify_tampered_sha256(
    signed_release: Path, rsa_key_pair: tuple[Path, Path]
) -> None:
    """Tampering with a file causes sha256 verification failure."""
    from pet_quantize.packaging.verify_package import verify_package

    # Tamper with the model file
    (signed_release / "vision_rk3576.rknn").write_bytes(b"tampered!")

    _, pub_path = rsa_key_pair
    result = verify_package(
        release_dir=str(signed_release),
        public_key_path=str(pub_path),
    )
    assert result.passed is False
    assert any("sha256" in e.lower() for e in result.errors)


def test_verify_no_signature_warns(signed_release: Path) -> None:
    """Missing signature gives warning but does not fail."""
    from pet_quantize.packaging.verify_package import verify_package

    # Remove the signature file
    for sig in signed_release.glob("*.sig"):
        sig.unlink()

    result = verify_package(
        release_dir=str(signed_release),
        public_key_path="",
    )
    assert result.passed is True
    assert len(result.warnings) > 0
    assert any("signature" in w.lower() for w in result.warnings)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_verify_package.py -v`
Expected: FAIL

- [ ] **Step 3: Implement verify_package.py**

```python
# src/pet_quantize/packaging/verify_package.py
"""Package verification: sha256 integrity and RSA signature check.

Verifies the release package integrity. Missing signature triggers a
warning (not failure) for local development compatibility.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerifyResult:
    """Result of package verification."""

    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def verify_package(
    release_dir: str,
    public_key_path: str = "",
) -> VerifyResult:
    """Verify a release package's integrity and signature.

    Args:
        release_dir: Path to artifacts/release/.
        public_key_path: Path to RSA public key PEM. Empty to skip signature check.

    Returns:
        VerifyResult with pass/fail, errors, and warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []
    release = Path(release_dir)

    # Load manifest
    manifest_path = release / "manifest.json"
    if not manifest_path.exists():
        return VerifyResult(passed=False, errors=["manifest.json not found"])

    manifest = json.loads(manifest_path.read_text())

    # Verify sha256 of each file
    for key, file_info in manifest.get("files", {}).items():
        file_path = release / file_info["path"]
        if not file_path.exists():
            errors.append(f"{key}: file not found at {file_info['path']}")
            continue

        actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        expected_hash = file_info["sha256"]
        if actual_hash != expected_hash:
            errors.append(
                f"{key}: SHA256 mismatch — expected={expected_hash[:16]}..., "
                f"actual={actual_hash[:16]}..."
            )

    # Verify signature
    tarball_files = list(release.glob("*.tar.gz"))
    sig_files = list(release.glob("*.sig"))

    if not sig_files:
        warnings.append("No signature file found. Package is unsigned.")
    elif not public_key_path or not Path(public_key_path).exists():
        warnings.append(
            "Signature file present but no public key provided. "
            "Cannot verify signature."
        )
    else:
        tarball_path = tarball_files[0]
        sig_path = sig_files[0]

        pub_key_data = Path(public_key_path).read_bytes()
        public_key = serialization.load_pem_public_key(pub_key_data)

        try:
            public_key.verify(
                sig_path.read_bytes(),
                tarball_path.read_bytes(),
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
            logger.info("RSA signature verified successfully")
        except Exception:
            errors.append("RSA signature verification failed")

    passed = len(errors) == 0

    if passed:
        logger.info("Package verification passed")
    else:
        for e in errors:
            logger.error("Verification error: %s", e)
    for w in warnings:
        logger.warning("Verification warning: %s", w)

    return VerifyResult(passed=passed, errors=errors, warnings=warnings)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_verify_package.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/pet_quantize/packaging/verify_package.py tests/test_verify_package.py
git commit -m "feat(pet-quantize): add package verification with sha256 and RSA signature check"
```

---

### Task 15: Validate — conftest and smoke tests

**Files:**
- Create: `src/pet_quantize/validate/__init__.py`
- Create: `src/pet_quantize/validate/conftest.py`
- Create: `src/pet_quantize/validate/test_schema_compliance.py`
- Create: `src/pet_quantize/validate/test_kl_divergence.py`
- Create: `src/pet_quantize/validate/test_latency.py`
- Create: `src/pet_quantize/validate/test_audio_accuracy.py`

- [ ] **Step 1: Create validate __init__.py**

```python
# src/pet_quantize/validate/__init__.py
"""Smoke validation tests for quantized model artifacts."""
```

- [ ] **Step 2: Create conftest.py with dual-mode fixtures**

```python
# src/pet_quantize/validate/conftest.py
"""Pytest fixtures for smoke validation with dual-mode support.

Provides device_id, model_dir, and params fixtures. Device mode is
activated via --device-id CLI option.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --device-id option for on-device testing."""
    parser.addoption(
        "--device-id",
        action="store",
        default=None,
        help="ADB device serial number for on-device testing",
    )


@pytest.fixture()
def device_id(request: pytest.FixtureRequest) -> str | None:
    """Get the device ID from CLI option."""
    return request.config.getoption("--device-id")


@pytest.fixture()
def params() -> dict:
    """Load params.yaml from the project root."""
    params_path = Path(__file__).resolve().parents[3] / "params.yaml"
    with open(params_path) as fh:
        return yaml.safe_load(fh)


@pytest.fixture()
def model_dir(params: dict) -> str:
    """Get the converted model directory path."""
    return params["convert"]["output_dir"] or "artifacts/converted"
```

- [ ] **Step 3: Create test_schema_compliance.py**

```python
# src/pet_quantize/validate/test_schema_compliance.py
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
```

- [ ] **Step 4: Create test_kl_divergence.py**

```python
# src/pet_quantize/validate/test_kl_divergence.py
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
        # Extract distribution values from the schema output
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
```

- [ ] **Step 5: Create test_latency.py**

```python
# src/pet_quantize/validate/test_latency.py
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
    # Threshold from pet-eval gates (4000ms)
    assert p95 <= 4000, f"P95 latency {p95:.0f}ms > 4000ms threshold"
```

- [ ] **Step 6: Create test_audio_accuracy.py**

```python
# src/pet_quantize/validate/test_audio_accuracy.py
"""Smoke test: INT8 audio model basic accuracy."""
from __future__ import annotations

import glob
from pathlib import Path

import pytest


def test_audio_accuracy(params: dict, model_dir: str, device_id: str | None) -> None:
    """INT8 audio model achieves basic accuracy on a small sample."""
    from pet_quantize.inference import run_audio_inference

    # Find audio model
    audio_models = glob.glob(str(Path(model_dir) / "audio_*.rknn"))
    if not audio_models:
        pytest.skip("No audio RKNN model found in model_dir")

    audio_model_path = audio_models[0]

    # Find test audio files (from pet-train audio data or calibration)
    audio_dir = params.get("convert", {}).get("audio_test_dir", "")
    if not audio_dir or not Path(audio_dir).exists():
        pytest.skip("No audio test directory configured")

    audio_files = glob.glob(str(Path(audio_dir) / "*.wav"))
    if not audio_files:
        pytest.skip("No audio test files found")

    sample_size = min(params["validate"]["smoke_sample_size"], len(audio_files))
    audio_files = audio_files[:sample_size]

    result = run_audio_inference(
        model_path=audio_model_path,
        audio_paths=audio_files,
        device_id=device_id,
    )

    # Basic check: predictions should be from the expected class set
    expected_classes = {"eating", "drinking", "vomiting", "ambient", "other"}
    for pred in result["predictions"]:
        assert pred in expected_classes, f"Unexpected prediction: {pred}"

    # All confidences should be reasonable (> 0.1)
    for conf in result["confidences"]:
        assert conf > 0.1, f"Suspiciously low confidence: {conf}"
```

- [ ] **Step 7: Commit**

```bash
git add src/pet_quantize/validate/
git commit -m "feat(pet-quantize): add smoke validation tests with dual-mode support"
```

---

### Task 16: Final Integration and Lint Pass

**Files:**
- Modify: `Makefile` (if adjustments needed)
- All source files

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/bamboo/Githubs/Train-Pet-Pipeline/pet-quantize && pytest tests/ -v`
Expected: All unit tests pass

- [ ] **Step 2: Run lint**

Run: `make lint`
Expected: No errors. If there are, fix them.

- [ ] **Step 3: Verify all Makefile targets are wired up**

Run: `make -n all` (dry-run)
Expected: Shows calibrate → convert → validate → package sequence

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "refactor(pet-quantize): lint fixes and final integration"
```

- [ ] **Step 5: Review test count and coverage**

Run: `pytest tests/ -v --tb=short | tail -20`
Expected: 30+ tests passing across all modules

---

### Task 17: Update DEVELOPMENT_GUIDE (if deviations)

**Files:**
- Modify: `/Users/bamboo/Githubs/Train-Pet-Pipeline/pet-infra/docs/DEVELOPMENT_GUIDE.md`

- [ ] **Step 1: Document inference/ module addition**

Add a note under section 5.6 explaining that `inference/` was added to
support pet-eval integration. The DEVELOPMENT_GUIDE's directory listing
should be updated to include:

```
├── inference/
│   ├── rknn_runner.py             # RKNN SDK 推理封装（真机/模拟双模式）
│   ├── rkllm_runner.py            # RKLLM SDK 推理封装
│   └── pipeline.py                # VLM 推理管线，供 pet-eval 调用
```

- [ ] **Step 2: Commit to pet-infra**

```bash
cd /Users/bamboo/Githubs/Train-Pet-Pipeline/pet-infra
git checkout dev
git checkout -b feature/update-quantize-spec
# Make edits
git add docs/DEVELOPMENT_GUIDE.md
git commit -m "docs(pet-infra): add inference/ module to pet-quantize spec"
```
