"""Audio CNN RKNN FP16 quantization plugin for RK3576.

Calls the preserved SDK wrapper ``pet_quantize.convert.convert_audio
.convert_audio_to_rknn`` which loads a PyTorch checkpoint, exports to
temporary ONNX, and builds an RKNN FP16 model. FP16 means no calibration
required.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pet_infra.registry import CONVERTERS
from pet_schema.model_card import EdgeArtifact, ModelCard, QuantConfig

# Top-level import is safe: pet_quantize.convert.convert_audio itself
# lazy-imports rknn.api inside convert_audio_to_rknn (see that module).
# Module-load therefore does NOT trip the rknn SDK requirement.
from pet_quantize.convert import convert_audio as _audio_mod


@CONVERTERS.register_module(name="audio_rknn_fp16")
class AudioRknnFp16Converter:
    """CONVERTERS plugin — audio CNN to RKNN FP16 for RK3576."""

    def __init__(
        self,
        target_platform: str = "rk3576",
        output_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Store conversion parameters; extra kwargs stored for future SDK passthrough."""
        self.target_platform = target_platform
        self.output_dir = Path(output_dir) if output_dir else Path(".cache/rknn-audio")
        self._extra = dict(kwargs)

    def run(self, input_card: ModelCard, recipe: Any) -> ModelCard:
        """Export audio CNN to ONNX then convert to RKNN FP16."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "audio_checkpoint": input_card.checkpoint_uri,
            "output_dir": str(self.output_dir),
        }

        rknn_path = _audio_mod.convert_audio_to_rknn(config=config)

        out = Path(rknn_path)
        sha = hashlib.sha256(out.read_bytes()).hexdigest()

        edge = EdgeArtifact(
            format="rknn",
            target_hardware=[self.target_platform],
            artifact_uri=str(out),
            sha256=sha,
            size_bytes=out.stat().st_size,
            input_shape={"mel_spectrogram": [1, 1, 64, 100]},
        )
        quant_cfg = QuantConfig(method="fp16")
        return input_card.model_copy(
            update={
                "edge_artifacts": [*input_card.edge_artifacts, edge],
                "quantization": quant_cfg,
            }
        )
