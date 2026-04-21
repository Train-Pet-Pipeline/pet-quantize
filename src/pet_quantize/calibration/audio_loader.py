"""Thin stub — returns calibration audio clip tensors. Wire up to real source in P6+."""
from __future__ import annotations

import torch


def load_calibration_clips(source_uri: str, num_samples: int) -> list[torch.Tensor]:
    """Return a list of (1, 64, 100) float tensors for calibration.

    Raises:
        NotImplementedError: Always. Wire up to pet-data frame DB in P6 or later.
    """
    raise NotImplementedError(
        "load_calibration_clips is a stub; wire up to pet-data frame DB in P6 or later"
    )
