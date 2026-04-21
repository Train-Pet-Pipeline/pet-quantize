"""Entry-point target for pet-infra's plugin discovery.

Imports pet-quantize plugin modules to trigger @CONVERTERS.register_module and
@DATASETS.register_module side-effects. SDK-gated clusters use try/except
ImportError and re-raise unless PET_ALLOW_MISSING_SDK=1 is set.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def register_all() -> None:
    """Import pet-quantize plugin modules to trigger registration side-effects."""
    try:
        import pet_infra  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "pet-quantize v2 requires pet-infra. Install via matrix row 2026.08."
        ) from e

    # Always-available (zero-dep)
    from pet_quantize.plugins.converters import noop  # noqa: F401

    # RKNN-gated cluster: converters + calibration datasets that feed them.
    # Datasets are grouped here (not always-available) because registering a
    # dataset plugin that feeds a missing converter has no value and incurs the
    # torch-import cost in CI runs where the SDK is absent.
    try:
        import rknn.api  # noqa: F401

        from pet_quantize.plugins.converters import (
            audio_rknn_fp16,  # noqa: F401
            vision_rknn_fp16,  # noqa: F401
        )
        from pet_quantize.plugins.datasets import (
            audio_calibration_subset,  # noqa: F401
            vision_calibration_subset,  # noqa: F401
        )
    except ImportError as exc:
        if not os.environ.get("PET_ALLOW_MISSING_SDK"):
            raise
        logger.warning("rknn SDK missing; gated plugins skipped: %s", exc)

    # RKLLM-gated cluster: vlm converter + vlm calibration dataset that feeds it.
    try:
        import rkllm.api  # noqa: F401

        from pet_quantize.plugins.converters import vlm_rkllm_w4a16  # noqa: F401
        from pet_quantize.plugins.datasets import vlm_calibration_subset  # noqa: F401
    except ImportError as exc:
        if not os.environ.get("PET_ALLOW_MISSING_SDK"):
            raise
        logger.warning("rkllm SDK missing; gated plugins skipped: %s", exc)
