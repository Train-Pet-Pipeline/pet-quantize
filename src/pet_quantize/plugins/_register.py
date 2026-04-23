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
    # pet-infra is a β peer-dep (not in pyproject.dependencies as of v2.1.0);
    # the guard is intentionally inside register_all so bare `import pet_quantize`
    # remains lightweight for IDE / static-analysis use (see DEV_GUIDE §11.3
    # "delayed-guard" variant). peer-dep-smoke.yml is the producer-side contract.
    try:
        import pet_infra  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "pet-quantize requires pet-infra to be installed first. "
            "Install via latest matrix row (pet-infra/docs/compatibility_matrix.yaml)."
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
