"""DATASETS plugin — writes a VLM calibration tensor batch to content-addressable cache.

Content-addressable cache key is sha256(modality|source_uri|num_samples). Downstream
CONVERTERS plugins consume the batch via ``intermediate_artifacts['calibration_batch_uri']``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from pet_infra.registry import DATASETS
from pet_schema.model_card import ModelCard

from pet_quantize.calibration import vlm_loader as _loader_mod


@DATASETS.register_module(name="vlm_calibration_subset")
class VlmCalibrationSubset:
    """Produce a (num_samples, 2048) int64 tensor batch; cache under cache_dir."""

    def __init__(
        self,
        source_uri: str,
        num_samples: int = 64,
        batch_size: int = 8,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Store plugin parameters; extra kwargs stored for introspection."""
        self.source_uri = source_uri
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache/calibration")
        self._extra = dict(kwargs)

    def _cache_key(self) -> str:
        """Derive a 16-char hex key from (modality, source_uri, num_samples)."""
        payload = f"vlm|{self.source_uri}|{self.num_samples}".encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def run(self, input_card: ModelCard, recipe: Any) -> ModelCard:
        """Load calibration pairs, stack into a batch tensor, and cache to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"{self._cache_key()}.pt"
        if not cache_path.exists():
            tensors = _loader_mod.load_calibration_pairs(self.source_uri, self.num_samples)
            batch = torch.stack(tensors)
            torch.save(batch, cache_path)
        return input_card.model_copy(
            update={
                "intermediate_artifacts": {
                    **input_card.intermediate_artifacts,
                    "calibration_batch_uri": str(cache_path),
                },
            }
        )
