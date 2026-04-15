"""RKLLM SDK wrapper for LLM inference.

Supports dual-mode: on-device (ADB) and simulated (PC).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from rkllm.api import RKLLMRuntime

logger = logging.getLogger(__name__)


class RKLLMRunner:
    """Wrapper for RKLLM model inference with dual-mode support.

    Args:
        model_path: Path to the .rkllm model file.
        target: Target platform (e.g., "rk3576"). None for PC simulation.
        device_id: ADB device serial number. Required when target is set.
    """

    def __init__(
        self,
        model_path: str,
        target: str | None = None,
        device_id: str | None = None,
    ) -> None:
        if not Path(model_path).exists():
            msg = f"RKLLM model not found: {model_path}"
            raise FileNotFoundError(msg)

        self._model_path = model_path
        self._target = target
        self._device_id = device_id
        self._runtime: RKLLMRuntime | None = None

    def init(self) -> None:
        """Initialize the RKLLM runtime.

        Raises:
            RuntimeError: If initialization fails.
        """
        kwargs: dict[str, Any] = {"model_path": self._model_path}
        if self._target and self._device_id:
            kwargs["target"] = self._target
            kwargs["device_id"] = self._device_id
            logger.info(
                "Initializing RKLLM on-device: target=%s, device=%s",
                self._target,
                self._device_id,
            )
        else:
            logger.info("Initializing RKLLM simulated runtime (PC)")

        self._runtime = RKLLMRuntime(**kwargs)

    def generate(
        self,
        prompt: str,
        visual_features: Any | None = None,
        max_tokens: int = 2048,
    ) -> tuple[str, float]:
        """Generate text from a prompt with optional visual features.

        Args:
            prompt: Text prompt for the LLM.
            visual_features: Visual encoder output features (numpy array).
            max_tokens: Maximum number of tokens to generate.

        Returns:
            Tuple of (generated text, elapsed time in milliseconds).

        Raises:
            RuntimeError: If runtime is not initialized.
        """
        if self._runtime is None:
            msg = "RKLLM runtime not initialized. Call init() first."
            raise RuntimeError(msg)

        start = time.perf_counter()
        output = self._runtime.generate(
            prompt=prompt,
            visual_features=visual_features,
            max_new_tokens=max_tokens,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        return output, elapsed_ms

    def release(self) -> None:
        """Release the RKLLM runtime resources."""
        if self._runtime is not None:
            self._runtime.release()
            self._runtime = None
