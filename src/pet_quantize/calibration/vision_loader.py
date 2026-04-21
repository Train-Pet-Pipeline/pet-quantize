"""Thin stub — returns calibration image tensors. Wire up to real source in P6+."""
from __future__ import annotations

import torch


def load_calibration_images(source_uri: str, num_samples: int) -> list[torch.Tensor]:
    """Return a list of (3, 224, 224) float tensors for calibration.

    Raises:
        NotImplementedError: Always. Wire up to pet-data frame DB in P6 or later.
    """
    raise NotImplementedError(
        "load_calibration_images is a stub; wire up to pet-data frame DB in P6 or later"
    )
