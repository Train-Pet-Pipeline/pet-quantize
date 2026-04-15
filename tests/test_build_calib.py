"""Tests for calibration dataset builder."""

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from pet_quantize.calibration.build_calib_dataset import build_calib_dataset
from pet_quantize.calibration.validate_calib import validate_calibration_dataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config_with_db(
    sample_params: dict[str, Any],
    db_path: Path,
    train_ids_path: str = "",
    gold_set_path: str = "",
) -> dict[str, Any]:
    """Return calibration config dict pointing at *db_path*."""
    cfg = dict(sample_params["calibration"])
    cfg["data_db_path"] = str(db_path)
    cfg["exclude"] = {
        "train_ids_path": train_ids_path,
        "gold_set_path": gold_set_path,
    }
    return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sample_respects_frame_count(
    sample_calib_db: Path,
    sample_params: dict[str, Any],
) -> None:
    """build_calib_dataset must return exactly frame_count frames."""
    config = _config_with_db(sample_params, sample_calib_db)
    frames = build_calib_dataset(config)
    assert len(frames) == config["frame_count"]


def test_sample_excludes_train_ids(
    sample_calib_db: Path,
    sample_params: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Frames whose IDs appear in train_ids_path must not be in the result."""
    # First fetch all frame_ids from the db
    conn = sqlite3.connect(sample_calib_db)
    cur = conn.cursor()
    cur.execute("SELECT frame_id FROM frames LIMIT 20")
    excluded = {str(row[0]) for row in cur.fetchall()}
    conn.close()

    exclude_file = tmp_path / "train_ids.txt"
    exclude_file.write_text("\n".join(excluded))

    config = _config_with_db(
        sample_params,
        sample_calib_db,
        train_ids_path=str(exclude_file),
    )
    frames = build_calib_dataset(config)
    result_ids = {f["frame_id"] for f in frames}
    assert result_ids.isdisjoint(excluded), "Some excluded train IDs appeared in result"


def test_sample_excludes_gold_set_ids(
    sample_calib_db: Path,
    sample_params: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Frames whose IDs appear in gold_set_path must not be in the result."""
    conn = sqlite3.connect(sample_calib_db)
    cur = conn.cursor()
    cur.execute("SELECT frame_id FROM frames LIMIT 10 OFFSET 50")
    excluded = {str(row[0]) for row in cur.fetchall()}
    conn.close()

    gold_file = tmp_path / "gold_set.txt"
    gold_file.write_text("\n".join(excluded))

    config = _config_with_db(
        sample_params,
        sample_calib_db,
        gold_set_path=str(gold_file),
    )
    frames = build_calib_dataset(config)
    result_ids = {f["frame_id"] for f in frames}
    assert result_ids.isdisjoint(excluded), "Some excluded gold-set IDs appeared in result"


def test_sample_satisfies_distribution(
    sample_calib_db: Path,
    sample_params: dict[str, Any],
) -> None:
    """The sampled frames must pass validate_calibration_dataset."""
    config = _config_with_db(sample_params, sample_calib_db)
    frames = build_calib_dataset(config)
    result = validate_calibration_dataset(frames, config)
    assert result.passed is True, f"Distribution check failed: {result.violations}"


def test_insufficient_pool_raises(
    sample_calib_db: Path,
    sample_params: dict[str, Any],
) -> None:
    """Requesting more frames than available in a bucket must raise ValueError."""
    import copy

    config = copy.deepcopy(sample_params["calibration"])
    config["data_db_path"] = str(sample_calib_db)
    config["frame_count"] = 10_000  # far more than the ~400 rows in sample_calib_db
    config["exclude"] = {"train_ids_path": "", "gold_set_path": ""}

    with pytest.raises(ValueError):
        build_calib_dataset(config)


def test_missing_db_raises(
    sample_params: dict[str, Any],
    tmp_path: Path,
) -> None:
    """A nonexistent database path must raise FileNotFoundError."""
    config = _config_with_db(sample_params, tmp_path / "does_not_exist.db")
    with pytest.raises(FileNotFoundError):
        build_calib_dataset(config)
