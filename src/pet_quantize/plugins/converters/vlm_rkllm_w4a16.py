"""VLM RKLLM W4A16 quantization plugin for RK3576.

Calls the preserved SDK wrapper ``pet_quantize.convert.convert_to_rkllm.
convert_llm_to_rkllm`` and appends the resulting artifact to
``ModelCard.edge_artifacts``. Consumes ``intermediate_artifacts
.calibration_batch_uri`` produced by an upstream DATASETS stage.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pet_infra.registry import CONVERTERS
from pet_schema.model_card import EdgeArtifact, ModelCard, QuantConfig


@CONVERTERS.register_module(name="vlm_rkllm_w4a16")
class VlmRkllmW4A16Converter:
    """CONVERTERS plugin — VLM to RKLLM W4A16 for RK3576."""

    def __init__(
        self,
        target_platform: str = "rk3576",
        quantized_dtype: str = "w4a16",
        output_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Store conversion parameters; extra kwargs forwarded to SDK wrapper."""
        self.target_platform = target_platform
        self.quantized_dtype = quantized_dtype
        self.output_dir = Path(output_dir) if output_dir else Path(".cache/rkllm")
        self._extra = dict(kwargs)

    def run(self, input_card: ModelCard, recipe: Any) -> ModelCard:
        """Quantize the VLM checkpoint to RKLLM and append EdgeArtifact to card.

        Requires ``input_card.intermediate_artifacts['calibration_batch_uri']``
        to be set by a preceding DATASETS stage.

        Note: ``calibration_batch_uri`` is passed through as-is to ``calib_dir``.
        P2-F datasets plugin will determine whether the URI points to a directory
        or file; tests mock the wrapper so the real shape does not block this PR.
        """
        cal_uri = input_card.intermediate_artifacts.get("calibration_batch_uri")
        if not cal_uri:
            raise ValueError(
                "VlmRkllmW4A16Converter requires card.intermediate_artifacts."
                "calibration_batch_uri (produced by an upstream DATASETS stage "
                "like vlm_calibration_subset)"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "weights_dir": input_card.checkpoint_uri,
            "output_dir": str(self.output_dir),
            "llm": {
                "rkllm_target": self.target_platform,
                "quantization": self.quantized_dtype,
            },
        }

        # Lazy import: convert_to_rkllm transitively imports rkllm.api at function
        # level, but pet-eval's test_module_load_does_not_import_rkllm_runner
        # assertion depends on this module being importable without rkllm even
        # loaded — keep the import inside run() to preserve that contract.
        import pet_quantize.convert.convert_to_rkllm as _rkllm_mod  # lazy (SDK-bound)

        output_path = _rkllm_mod.convert_llm_to_rkllm(config=config, calib_dir=cal_uri)
        out = Path(output_path)
        sha = hashlib.sha256(out.read_bytes()).hexdigest()

        edge = EdgeArtifact(
            format="rkllm",
            target_hardware=[self.target_platform],
            artifact_uri=str(out),
            sha256=sha,
            size_bytes=out.stat().st_size,
            input_shape={"input_ids": [1, 2048]},
        )
        quant_cfg = QuantConfig(
            method="ptq_int8",
            bits=4 if "4" in self.quantized_dtype else 8,
            calibration_dataset_uri=cal_uri,
        )
        return input_card.model_copy(
            update={
                "edge_artifacts": [*input_card.edge_artifacts, edge],
                "quantization": quant_cfg,
            }
        )
