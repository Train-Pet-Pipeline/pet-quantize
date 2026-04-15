"""Configuration models and utilities for pet-quantize.

Loads and validates params.yaml via Pydantic, and configures structured JSON logging.
All numeric values are read from params.yaml — never hardcoded here.
"""

import logging
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pythonjsonlogger import jsonlogger

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CalibDistribution(BaseModel):
    """Target sampling distribution for calibration frames."""

    lighting: dict[str, float]
    action_primary: dict[str, float]


class CalibExclude(BaseModel):
    """Paths to exclusion lists for calibration frame selection."""

    train_ids_path: str = ""
    gold_set_path: str = ""


class CalibrationConfig(BaseModel):
    """Parameters for the calibration dataset builder."""

    frame_count: int
    tolerance: float
    min_breeds: int
    distribution: CalibDistribution
    exclude: CalibExclude
    data_db_path: str
    output_dir: str


class VisionConvertConfig(BaseModel):
    """Conversion settings for the vision encoder."""

    input_size: list[int]
    onnx_opset: int
    rknn_target: str
    rknn_dtype: str


class LlmConvertConfig(BaseModel):
    """Conversion settings for the on-device LLM."""

    rkllm_target: str
    quantization: str


class AudioConvertConfig(BaseModel):
    """Conversion settings for the audio CNN."""

    rknn_target: str
    rknn_dtype: str


class ConvertConfig(BaseModel):
    """Top-level conversion configuration."""

    vision: VisionConvertConfig
    llm: LlmConvertConfig
    audio: AudioConvertConfig
    weights_dir: str
    audio_checkpoint: str
    output_dir: str


class DeviceConfig(BaseModel):
    """ADB device and latency-measurement settings."""

    adb_timeout: int
    warmup_runs: int
    latency_runs: int


class InferenceConfig(BaseModel):
    """Parameters for the simulated / on-device inference runner."""

    schema_version: str
    simulated_sample_size: int
    device: DeviceConfig
    fp16_weights_dir: str


class ValidateConfig(BaseModel):
    """Parameters for smoke-test validation."""

    smoke_sample_size: int
    kl_threshold: float


class PackagingConfig(BaseModel):
    """Artifact packaging metadata."""

    version: str
    lora_version: str
    min_firmware: str
    release_notes: str


class WandbConfig(BaseModel):
    """Weights & Biases project settings."""

    project: str
    entity: str


class QuantizeParams(BaseModel):
    """Root model representing the full params.yaml for pet-quantize."""

    model_config = ConfigDict(populate_by_name=True)

    calibration: CalibrationConfig
    convert: ConvertConfig
    inference: InferenceConfig
    validate_cfg: ValidateConfig = Field(alias="validate")
    packaging: PackagingConfig
    wandb: WandbConfig


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def load_params(path: Path) -> QuantizeParams:
    """Load and validate params.yaml from *path*.

    Args:
        path: Filesystem path to the YAML params file.

    Returns:
        A fully-validated :class:`QuantizeParams` instance.

    Raises:
        FileNotFoundError: When *path* does not exist.
        pydantic.ValidationError: When the YAML content fails schema validation.
    """
    if not path.exists():
        raise FileNotFoundError(f"params file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    return QuantizeParams.model_validate(raw)


def setup_logging() -> None:
    """Configure root logger to emit structured JSON log records.

    Idempotent: if a :class:`~pythonjsonlogger.jsonlogger.JsonFormatter`-backed
    handler is already attached to the root logger, no additional handler is added.
    """
    root = logging.getLogger()

    # Guard: skip if a JSON handler is already present
    for handler in root.handlers:
        if isinstance(handler.formatter, jsonlogger.JsonFormatter):
            return

    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(logging.INFO)
