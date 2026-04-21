"""Vision RKNN FP16 quantization plugin for RK3576.

Chains two preserved SDK wrappers:
1. ``pet_quantize.convert.export_vision_encoder.export_vision_encoder`` —
   PyTorch checkpoint → ONNX.
2. ``pet_quantize.convert.convert_to_rknn.convert_vision_to_rknn`` —
   ONNX → RKNN FP16.

FP16 means no post-training calibration, so ``intermediate_artifacts
.calibration_batch_uri`` is NOT required.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pet_infra.registry import CONVERTERS
from pet_schema.model_card import EdgeArtifact, ModelCard, QuantConfig

from pet_quantize.convert import convert_to_rknn as _rknn_mod
from pet_quantize.convert import export_vision_encoder as _export_mod


@CONVERTERS.register_module(name="vision_rknn_fp16")
class VisionRknnFp16Converter:
    """CONVERTERS plugin — vision encoder to RKNN FP16 for RK3576."""

    def __init__(
        self,
        target_platform: str = "rk3576",
        optimization_level: int = 3,
        output_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Store conversion parameters; extra kwargs forwarded to SDK wrapper."""
        self.target_platform = target_platform
        self.optimization_level = optimization_level
        self.output_dir = Path(output_dir) if output_dir else Path(".cache/rknn")
        self._extra = dict(kwargs)

    def run(self, input_card: ModelCard, recipe: Any) -> ModelCard:
        """Export vision encoder to ONNX then convert to RKNN FP16."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "weights_dir": input_card.checkpoint_uri,
            "output_dir": str(self.output_dir),
            "vision": {
                "rknn_target": self.target_platform,
                "optimization_level": self.optimization_level,
            },
        }

        onnx_path = _export_mod.export_vision_encoder(config=config)
        rknn_path = _rknn_mod.convert_vision_to_rknn(onnx_path=onnx_path, config=config)

        out = Path(rknn_path)
        sha = hashlib.sha256(out.read_bytes()).hexdigest()

        edge = EdgeArtifact(
            format="rknn",
            target_hardware=[self.target_platform],
            artifact_uri=str(out),
            sha256=sha,
            size_bytes=out.stat().st_size,
            input_shape={"pixel_values": [1, 3, 448, 448]},
        )
        quant_cfg = QuantConfig(method="fp16")
        return input_card.model_copy(
            update={
                "edge_artifacts": [*input_card.edge_artifacts, edge],
                "quantization": quant_cfg,
                "intermediate_artifacts": {
                    **input_card.intermediate_artifacts,
                    "vision_onnx_uri": onnx_path,
                },
            }
        )
