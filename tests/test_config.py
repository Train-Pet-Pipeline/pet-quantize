"""Tests for pet_quantize.config module."""

import json
import logging
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from pet_quantize.config import QuantizeParams, load_params, setup_logging


class TestLoadParams:
    """Tests for load_params()."""

    def test_load_params_valid(self, sample_params_path: Path) -> None:
        """loads valid params file and checks key fields."""
        params = load_params(sample_params_path)

        assert isinstance(params, QuantizeParams)
        # calibration
        assert params.calibration.frame_count == 200
        assert params.calibration.tolerance == 0.05
        assert params.calibration.min_breeds == 5
        assert params.calibration.distribution.lighting["bright"] == pytest.approx(0.40)
        assert params.calibration.distribution.action_primary["eating"] == pytest.approx(0.50)
        assert params.calibration.exclude.train_ids_path == ""
        assert params.calibration.exclude.gold_set_path == ""
        # convert
        assert params.convert.vision.input_size == [448, 448]
        assert params.convert.vision.onnx_opset == 17
        assert params.convert.vision.rknn_target == "rk3576"
        assert params.convert.vision.rknn_dtype == "fp16"
        assert params.convert.llm.rkllm_target == "rk3576"
        assert params.convert.llm.quantization == "w8a8"
        assert params.convert.audio.rknn_target == "rk3576"
        assert params.convert.audio.rknn_dtype == "int8"
        # inference
        assert params.inference.schema_version == "1.0"
        assert params.inference.simulated_sample_size == 50
        assert params.inference.device.adb_timeout == 30
        assert params.inference.device.warmup_runs == 3
        assert params.inference.device.latency_runs == 20
        # validate
        assert params.validate_cfg.smoke_sample_size == 30
        assert params.validate_cfg.kl_threshold == pytest.approx(0.05)
        # packaging
        assert params.packaging.version == "1.0.0"
        assert params.packaging.lora_version == "1.0"
        assert params.packaging.min_firmware == "2.0.0"
        assert params.packaging.release_notes == "Initial release"
        # wandb
        assert params.wandb.project == "pet-quantize"
        assert params.wandb.entity == ""

    def test_load_params_missing_field(self, tmp_dir: Path) -> None:
        """Missing a required section raises pydantic ValidationError."""
        incomplete = {
            "calibration": {
                "frame_count": 200,
                "tolerance": 0.05,
                "min_breeds": 5,
                "distribution": {
                    "lighting": {"bright": 1.0},
                    "action_primary": {"eating": 1.0},
                },
                "exclude": {"train_ids_path": "", "gold_set_path": ""},
                "data_db_path": "",
                "output_dir": "",
            }
            # 'convert', 'inference', 'validate', 'packaging', 'wandb' are intentionally absent
        }
        path = tmp_dir / "bad_params.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(incomplete, f)

        with pytest.raises(ValidationError):
            load_params(path)

    def test_load_params_file_not_found(self, tmp_dir: Path) -> None:
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_params(tmp_dir / "nonexistent.yaml")


class TestSetupLogging:
    """Tests for setup_logging()."""

    def test_setup_logging_json_format(self, capfd: pytest.CaptureFixture[str]) -> None:
        """setup_logging configures JSON output that can be parsed."""
        # Reset root logger handlers before testing
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        for h in root.handlers[:]:
            root.removeHandler(h)

        try:
            setup_logging()
            logger = logging.getLogger("pet_quantize.test_json")
            logger.info("test message", extra={"key": "value"})

            captured = capfd.readouterr()
            output = captured.err + captured.out
            # Find a line that looks like JSON
            json_line = None
            for line in output.splitlines():
                line = line.strip()
                if line.startswith("{"):
                    try:
                        json_line = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue

            assert json_line is not None, f"No JSON log line found in output: {output!r}"
            assert "message" in json_line or "msg" in json_line
        finally:
            # Restore original handlers
            for h in root.handlers[:]:
                root.removeHandler(h)
            for h in original_handlers:
                root.addHandler(h)

    def test_setup_logging_idempotent(self) -> None:
        """Calling setup_logging twice does not duplicate JSON handlers."""
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        for h in root.handlers[:]:
            root.removeHandler(h)

        try:
            setup_logging()
            count_after_first = len(root.handlers)

            setup_logging()
            count_after_second = len(root.handlers)

            assert count_after_second == count_after_first, (
                f"Handlers duplicated: {count_after_first} -> {count_after_second}"
            )
        finally:
            for h in root.handlers[:]:
                root.removeHandler(h)
            for h in original_handlers:
                root.addHandler(h)
