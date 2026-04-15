"""Calibration dataset builder.

Queries frames from a SQLite database, filters excluded IDs, and produces a
distribution-aware stratified sample for quantization calibration.
All configuration values are read from the caller-supplied config dict.
"""

import logging
import random
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_exclude_ids(path: str) -> set[str]:
    """Load newline-delimited frame IDs to exclude from *path*.

    Args:
        path: Filesystem path to the exclusion list, or empty string to skip.

    Returns:
        Set of string frame IDs.  Returns an empty set when *path* is empty or
        the file does not exist (the latter also emits a WARNING log entry).
    """
    if not path:
        return set()

    p = Path(path)
    if not p.exists():
        logger.warning("exclude-ids file not found, skipping", extra={"path": path})
        return set()

    ids: set[str] = set()
    for line in p.read_text().splitlines():
        stripped = line.strip()
        if stripped:
            ids.add(stripped)

    return ids


def _query_frames(db_path: str, exclude_ids: set[str]) -> list[dict[str, str]]:
    """Query all frames from *db_path* and filter out *exclude_ids*.

    Args:
        db_path: Path to the SQLite database.
        exclude_ids: Frame IDs (as strings) to exclude from the result.

    Returns:
        List of frame dicts with keys ``frame_id``, ``image_path``,
        ``lighting``, ``action_primary``, and ``breed``.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT frame_id, image_path, lighting, action_primary, breed FROM frames"
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    frames: list[dict[str, str]] = []
    for row in rows:
        fid = str(row["frame_id"])
        if fid in exclude_ids:
            continue
        frames.append(
            {
                "frame_id": fid,
                "image_path": str(row["image_path"]),
                "lighting": str(row["lighting"]),
                "action_primary": str(row["action_primary"]),
                "breed": str(row["breed"]),
            }
        )

    return frames


def _stratified_sample(
    frames: list[dict[str, str]],
    target_count: int,
    distribution: dict[str, Any],
) -> list[dict[str, str]]:
    """Return a stratified random sample of *target_count* frames.

    Groups *frames* by ``(lighting, action_primary)`` bucket.  For each
    bucket the target number of frames is computed as::

        target_count × lighting[L] × action_primary[A]

    Rounding errors across all buckets are corrected by adjusting the
    largest bucket so the total equals exactly *target_count*.

    Args:
        frames: Full candidate pool.
        target_count: Desired sample size.
        distribution: Dict with keys ``"lighting"`` and ``"action_primary"``,
                      each mapping category → target proportion.

    Returns:
        Sampled list of frame dicts.

    Raises:
        ValueError: When any bucket has fewer frames than its target count.
    """
    lighting_dist: dict[str, float] = distribution["lighting"]
    action_dist: dict[str, float] = distribution["action_primary"]

    # Group frames by (lighting, action_primary)
    buckets: dict[tuple[str, str], list[dict[str, str]]] = {}
    for frame in frames:
        key = (frame["lighting"], frame["action_primary"])
        buckets.setdefault(key, []).append(frame)

    # Compute per-bucket target counts using joint probability
    bucket_targets: dict[tuple[str, str], int] = {}
    for lit_cat, lit_prop in lighting_dist.items():
        for act_cat, act_prop in action_dist.items():
            key = (lit_cat, act_cat)
            bucket_targets[key] = round(target_count * lit_prop * act_prop)

    # Fix rounding: ensure total == target_count
    total_rounded = sum(bucket_targets.values())
    diff = target_count - total_rounded
    if diff != 0:
        # Adjust the bucket with the largest target count
        largest_key = max(bucket_targets, key=lambda k: bucket_targets[k])
        bucket_targets[largest_key] += diff

    # Validate availability and sample
    sampled: list[dict[str, str]] = []
    for key, needed in bucket_targets.items():
        if needed == 0:
            continue
        available = buckets.get(key, [])
        if len(available) < needed:
            lit_cat, act_cat = key
            raise ValueError(
                f"Insufficient frames for bucket ({lit_cat}, {act_cat}): "
                f"needed={needed} available={len(available)}"
            )
        sampled.extend(random.sample(available, needed))

    return sampled


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_calib_dataset(config: dict[str, Any]) -> list[dict[str, str]]:
    """Build a calibration dataset from the parameters in *config*.

    Orchestration:
    1. Verify the database file exists.
    2. Load exclusion ID lists (train set and gold set).
    3. Query frames from the SQLite database, filtering excluded IDs.
    4. Stratified-sample the requested number of frames according to the
       target distribution.

    Args:
        config: The ``calibration`` section of params.yaml as a plain dict.

    Returns:
        List of sampled frame dicts.

    Raises:
        FileNotFoundError: When the SQLite database file does not exist.
        ValueError: When a distribution bucket has insufficient frames.
    """
    db_path = config["data_db_path"]
    if not Path(db_path).exists():
        raise FileNotFoundError(f"calibration database not found: {db_path}")

    exclude_cfg: dict[str, str] = config.get("exclude", {})
    train_ids = _load_exclude_ids(exclude_cfg.get("train_ids_path", ""))
    gold_ids = _load_exclude_ids(exclude_cfg.get("gold_set_path", ""))
    exclude_ids = train_ids | gold_ids

    frames = _query_frames(db_path, exclude_ids)
    logger.info(
        "queried frames from DB",
        extra={"total": len(frames) + len(exclude_ids), "after_exclusion": len(frames)},
    )

    frame_count: int = config["frame_count"]
    distribution: dict[str, Any] = config["distribution"]
    sampled = _stratified_sample(frames, frame_count, distribution)

    logger.info("calibration dataset built", extra={"sampled": len(sampled)})
    return sampled
