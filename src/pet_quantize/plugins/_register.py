"""Entry-point target for pet-infra's plugin discovery.

Imports pet-quantize plugin modules to trigger @CONVERTERS.register_module and
@DATASETS.register_module side-effects. SDK-gated clusters use try/except
ImportError and re-raise unless PET_ALLOW_MISSING_SDK=1 is set.
"""
from __future__ import annotations

import logging

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

    # RKNN-gated cluster (populated by P2-D/P2-E/P2-F)
    # P2-C+: from pet_quantize.plugins.converters import vision_rknn_fp16, audio_rknn_fp16
    # P2-C+: from pet_quantize.plugins.datasets import vision_calibration, audio_calibration

    # RKLLM-gated cluster (populated by P2-C/P2-F)
    # P2-C+: from pet_quantize.plugins.converters import vlm_rkllm_w4a16
    # P2-C+: from pet_quantize.plugins.datasets import vlm_calibration
