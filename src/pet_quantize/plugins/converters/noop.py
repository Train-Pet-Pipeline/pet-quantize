"""Zero-dependency CONVERTERS plugin used by PR CI with PET_ALLOW_MISSING_SDK=1.

Produces a deterministic fake EdgeArtifact so downstream stages have something
to consume without invoking any vendor SDK. Keyed on input_card.id so the
output is reproducible across runs (supports orchestrator resume-from-cache).
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

from pet_infra.registry import CONVERTERS
from pet_schema.model_card import EdgeArtifact, ModelCard


@CONVERTERS.register_module(name="noop_converter")
class NoopConverter:
    """Deterministic no-op converter for PR CI and fast smoke recipes."""

    def __init__(self, **kwargs: Any) -> None:
        """Accept any kwargs from RecipeStage.config; stored for introspection."""
        self._cfg = dict(kwargs)

    def run(self, input_card: ModelCard, recipe: Any) -> ModelCard:
        """Append a synthetic EdgeArtifact derived from input_card.id."""
        out_dir = Path(tempfile.mkdtemp(prefix="noop-artifact-"))
        artifact_path = out_dir / "model.noop"
        artifact_bytes = f"noop:{input_card.id}".encode()
        artifact_path.write_bytes(artifact_bytes)
        sha = hashlib.sha256(artifact_bytes).hexdigest()
        edge = EdgeArtifact(
            format="onnx",  # PR-CI-only placeholder; not a real ONNX file
            target_hardware=["cpu"],
            artifact_uri=str(artifact_path),
            sha256=sha,
            size_bytes=len(artifact_bytes),
            input_shape={"input": [1, 3, 224, 224]},
        )
        return input_card.model_copy(update={"edge_artifacts": [*input_card.edge_artifacts, edge]})
