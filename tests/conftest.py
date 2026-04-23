"""Shared pytest fixtures for pet-quantize tests."""

import sqlite3
from pathlib import Path
from typing import Any

import pytest
import yaml


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for the test."""
    return tmp_path


@pytest.fixture()
def sample_params() -> dict[str, Any]:
    """Minimal params.yaml dict with all sections."""
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
            "exclude": {
                "train_ids_path": "",
                "gold_set_path": "",
            },
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
            "llm": {
                "rkllm_target": "rk3576",
                "quantization": "w8a8",
            },
            "audio": {
                "rknn_target": "rk3576",
                "rknn_dtype": "int8",
            },
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
        "validate": {
            "smoke_sample_size": 30,
            "kl_threshold": 0.05,
        },
        "packaging": {
            "version": "1.0.0",
            "lora_version": "1.0",
            "min_firmware": "2.0.0",
            "release_notes": "Initial release",
        },
        "gates": {
            "vlm": {
                "schema_compliance": 0.99,
                "distribution_sum_error": 0.01,
                "latency_p95_ms": 4000,
                "kl_divergence": 0.02,
            },
            "audio": {
                "overall_accuracy": 0.80,
                "vomit_recall": 0.70,
            },
        },
        "audio": {
            "sample_rate": 16000,
            "n_mels": 64,
            "classes": ["eating", "drinking", "vomiting", "ambient", "other"],
        },
    }


@pytest.fixture()
def sample_params_path(tmp_dir: Path, sample_params: dict[str, Any]) -> Path:
    """Write sample_params to a yaml file and return its path."""
    path = tmp_dir / "params.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(sample_params, f)
    return path


@pytest.fixture()
def sample_calib_db(tmp_dir: Path) -> Path:
    """Create a SQLite DB with a frames table populated with enough rows for sampling."""
    db_path = tmp_dir / "calib.db"

    lighting_values = ["bright", "dim", "infrared_night", "unknown"]
    action_values = ["eating", "sniffing_only", "leaving_bowl", "other"]
    breeds = ["persian", "siamese", "maine_coon", "bengal", "ragdoll", "sphynx"]

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE frames (
            frame_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path  TEXT NOT NULL,
            lighting    TEXT NOT NULL,
            action_primary TEXT NOT NULL,
            breed       TEXT NOT NULL
        )
        """
    )

    rows = []
    frame_id = 1
    # ~400 rows: cover all 16 (lighting × action_primary) combinations with enough repetition.
    # Each of the 16 buckets gets ~25 frames; breeds cycle across all rows.
    breed_idx = 0
    for lighting in lighting_values:
        for action in action_values:
            for _ in range(60):
                breed = breeds[breed_idx % len(breeds)]
                image_path = f"/data/frames/frame_{frame_id:05d}.jpg"
                rows.append((image_path, lighting, action, breed))
                frame_id += 1
                breed_idx += 1

    cur.executemany(
        "INSERT INTO frames (image_path, lighting, action_primary, breed) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    return db_path
