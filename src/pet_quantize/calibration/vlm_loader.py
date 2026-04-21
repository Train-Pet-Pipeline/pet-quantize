"""Thin stub — returns calibration VLM input tensors. Wire up to real source in P6+."""
from __future__ import annotations

import torch


def load_calibration_pairs(source_uri: str, num_samples: int) -> list[torch.Tensor]:
    """Return a list of (2048,) int64 tensors (token_ids) for calibration.

    Raises:
        NotImplementedError: Always. Wire up to pet-data frame DB in P6 or later.
    """
    raise NotImplementedError(
        "load_calibration_pairs is a stub; wire up to pet-data frame DB in P6 or later"
    )
