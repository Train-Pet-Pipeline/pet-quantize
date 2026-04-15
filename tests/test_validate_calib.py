"""Tests for calibration distribution validation."""

from typing import Any

from pet_quantize.calibration.validate_calib import (
    CalibValidationResult,
    validate_calibration_dataset,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(frame_id: str, lighting: str, action: str, breed: str) -> dict[str, str]:
    """Return a minimal frame dict."""
    return {
        "frame_id": frame_id,
        "lighting": lighting,
        "action_primary": action,
        "breed": breed,
    }


def _make_valid_frames(count: int = 200) -> list[dict[str, str]]:
    """Generate exactly *count* frames matching the target distribution.

    Target joint distribution (lighting × action_primary):
        bright/eating=40, bright/sniffing_only=16, bright/leaving_bowl=12, bright/other=12
        dim/eating=20,    dim/sniffing_only=8,    dim/leaving_bowl=6,    dim/other=6
        infrared_night matches dim pattern (20+8+6+6=40)
        unknown          matches dim pattern (20+8+6+6=40)

    Total: 80+40+40+40 = 200 frames.
    5 breeds cycling.
    """
    breeds = ["persian", "siamese", "maine_coon", "bengal", "ragdoll"]

    buckets: list[tuple[str, str, int]] = [
        # (lighting, action, count)
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

    frames: list[dict[str, str]] = []
    idx = 0
    for lighting, action, n in buckets:
        for i in range(n):
            breed = breeds[idx % len(breeds)]
            frames.append(_make_frame(f"f{idx:04d}", lighting, action, breed))
            idx += 1

    assert len(frames) == count, f"Expected {count} frames, got {len(frames)}"
    return frames


def _make_config() -> dict[str, Any]:
    """Return the standard calibration config matching params.yaml."""
    return {
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
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_valid_distribution_passes() -> None:
    """A perfectly balanced dataset must pass all checks."""
    frames = _make_valid_frames()
    config = _make_config()
    result = validate_calibration_dataset(frames, config)
    assert isinstance(result, CalibValidationResult)
    assert result.passed is True
    assert result.violations == []


def test_lighting_proportion_violation() -> None:
    """Swapping 20 dim frames to bright pushes bright proportion above tolerance."""
    frames = _make_valid_frames()
    config = _make_config()
    # Swap 20 dim→bright (dim goes from 40→20 frames, bright goes from 80→100 frames)
    dim_indices = [i for i, f in enumerate(frames) if f["lighting"] == "dim"]
    for i in dim_indices[:20]:
        frames[i] = dict(frames[i], lighting="bright")
    result = validate_calibration_dataset(frames, config)
    assert result.passed is False
    assert any("lighting" in v for v in result.violations)


def test_action_proportion_violation() -> None:
    """Replacing all leaving_bowl frames with eating pushes eating above tolerance."""
    frames = _make_valid_frames()
    config = _make_config()
    frames = [
        (dict(f, action_primary="eating") if f["action_primary"] == "leaving_bowl" else f)
        for f in frames
    ]
    result = validate_calibration_dataset(frames, config)
    assert result.passed is False
    assert any("action_primary" in v for v in result.violations)


def test_insufficient_breeds() -> None:
    """Dataset with only 3 distinct breeds must fail min_breeds check."""
    frames = _make_valid_frames()
    config = _make_config()
    allowed_breeds = {"persian", "siamese", "maine_coon"}
    replacement_breeds = list(allowed_breeds)
    frames = [
        dict(f, breed=replacement_breeds[i % len(replacement_breeds)])
        for i, f in enumerate(frames)
    ]
    result = validate_calibration_dataset(frames, config)
    assert result.passed is False
    assert any("breed" in v for v in result.violations)


def test_wrong_frame_count() -> None:
    """Passing 100 frames when config expects 200 must fail."""
    frames = _make_valid_frames()[:100]
    config = _make_config()
    result = validate_calibration_dataset(frames, config)
    assert result.passed is False
    assert any("frame_count" in v for v in result.violations)


def test_boundary_tolerance_passes() -> None:
    """Shifting bright to exactly +5% (10 frames unknown→bright) must still pass.

    Base bright = 80/200 = 40.0 %.
    Move 10 frames from unknown to bright → bright = 90/200 = 45.0 % = target + 5.0 %.
    Tolerance is 5 %, so this is exactly at the boundary and must pass (≤ tolerance).
    """
    frames = _make_valid_frames()
    config = _make_config()
    unknown_indices = [i for i, f in enumerate(frames) if f["lighting"] == "unknown"]
    for i in unknown_indices[:10]:
        frames[i] = dict(frames[i], lighting="bright")
    result = validate_calibration_dataset(frames, config)
    assert result.passed is True, f"Expected pass at boundary; violations: {result.violations}"


def test_boundary_tolerance_fails() -> None:
    """Shifting bright beyond +5% (12 frames unknown→bright) must fail.

    bright = 92/200 = 46.0 % > target 40 % + 5 % tolerance → violation.
    """
    frames = _make_valid_frames()
    config = _make_config()
    unknown_indices = [i for i, f in enumerate(frames) if f["lighting"] == "unknown"]
    for i in unknown_indices[:12]:
        frames[i] = dict(frames[i], lighting="bright")
    result = validate_calibration_dataset(frames, config)
    assert result.passed is False
    assert any("lighting" in v for v in result.violations)
