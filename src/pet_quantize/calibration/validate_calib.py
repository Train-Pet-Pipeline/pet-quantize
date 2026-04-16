"""Calibration dataset distribution validation.

Checks that a list of frame dicts satisfies the target distribution
defined in params.yaml (frame count, lighting, action_primary, breed count).
All thresholds are read from the supplied config dict — never hardcoded.
"""

from dataclasses import dataclass, field
from typing import Any

import yaml
from pet_infra.logging import setup_logging

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Guards against floating-point rounding errors when checking exact boundary tolerance
_FP_EPSILON: float = 1e-9

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibValidationResult:
    """Immutable result of a calibration dataset validation run.

    Attributes:
        passed: True when all checks pass (violations list is empty).
        violations: Human-readable descriptions of each failed check.
    """

    passed: bool
    violations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_distribution(
    frames: list[dict[str, str]],
    field_name: str,
    target_dict: dict[str, float],
    tolerance: float,
) -> list[str]:
    """Check that the actual proportions of *field_name* match *target_dict* within *tolerance*.

    Args:
        frames: List of frame dicts each containing a key equal to *field_name*.
        field_name: The frame dict key whose distribution is being checked
                    (e.g. ``"lighting"`` or ``"action_primary"``).
        target_dict: Mapping of category → target proportion (0.0–1.0).
        tolerance: Allowed absolute deviation from the target proportion.

    Returns:
        A list of violation strings (empty list if all checks pass).
    """
    total = len(frames)
    if total == 0:
        return [f"{field_name}: no frames to check distribution"]

    # Count occurrences of each category
    counts: dict[str, int] = {}
    for frame in frames:
        category = frame.get(field_name, "")
        counts[category] = counts.get(category, 0) + 1

    violations: list[str] = []
    for category, target_prop in target_dict.items():
        actual_prop = counts.get(category, 0) / total
        deviation = abs(actual_prop - target_prop)
        if deviation > tolerance + _FP_EPSILON:
            violations.append(
                f"{field_name}[{category}]: actual={actual_prop:.4f} "
                f"target={target_prop:.4f} deviation={deviation:.4f} > tolerance={tolerance}"
            )

    return violations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_calibration_dataset(
    frames: list[dict[str, str]],
    config: dict[str, Any],
) -> CalibValidationResult:
    """Validate that *frames* satisfy all distribution constraints in *config*.

    Checks performed (in order):
    1. Frame count equals ``config["frame_count"]``.
    2. Lighting distribution within ``config["tolerance"]`` of targets.
    3. ``action_primary`` distribution within tolerance.
    4. Number of distinct breeds >= ``config["min_breeds"]``.

    Args:
        frames: Calibration frame dicts, each with keys
                ``frame_id``, ``lighting``, ``action_primary``, ``breed``.
        config: The ``calibration`` section of params.yaml as a plain dict.

    Returns:
        A :class:`CalibValidationResult` with ``passed=True`` when all checks
        succeed, or ``passed=False`` with a non-empty ``violations`` list.
    """
    violations: list[str] = []

    frame_count: int = config["frame_count"]
    tolerance: float = config["tolerance"]
    min_breeds: int = config["min_breeds"]
    distribution: dict[str, Any] = config["distribution"]

    # 1. Frame count
    if len(frames) != frame_count:
        violations.append(
            f"frame_count: expected={frame_count} actual={len(frames)}"
        )

    # 2. Lighting distribution
    lighting_target: dict[str, float] = distribution["lighting"]
    violations.extend(_check_distribution(frames, "lighting", lighting_target, tolerance))

    # 3. action_primary distribution
    action_target: dict[str, float] = distribution["action_primary"]
    violations.extend(_check_distribution(frames, "action_primary", action_target, tolerance))

    # 4. Breed count
    distinct_breeds = {f.get("breed", "") for f in frames}
    distinct_breeds.discard("")
    if len(distinct_breeds) < min_breeds:
        violations.append(
            f"breed count: expected>={min_breeds} actual={len(distinct_breeds)}"
        )

    return CalibValidationResult(passed=len(violations) == 0, violations=violations)


def main() -> None:
    """CLI entry point for calibration validation."""
    import logging

    setup_logging("pet-quantize")
    logger = logging.getLogger(__name__)

    with open("params.yaml") as fh:
        params = yaml.safe_load(fh)

    from pet_quantize.calibration.build_calib_dataset import build_calib_dataset
    frames = build_calib_dataset(params["calibration"])
    result = validate_calibration_dataset(frames, params["calibration"])

    if result.passed:
        logger.info("Calibration validation PASSED")
    else:
        for v in result.violations:
            logger.error("Calibration violation: %s", v)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
