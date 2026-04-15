"""RKNN SDK wrapper for vision encoder and audio model inference.

Supports dual-mode: on-device (ADB) and simulated (PC simulator).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from rknn.api import RKNN

logger = logging.getLogger(__name__)


class RKNNRunner:
    """Wrapper for RKNN model inference with dual-mode support.

    Args:
        model_path: Path to the .rknn model file.
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
            msg = f"RKNN model not found: {model_path}"
            raise FileNotFoundError(msg)

        self._model_path = model_path
        self._target = target
        self._device_id = device_id
        self._rknn: RKNN | None = None

    def init(self) -> None:
        """Initialize the RKNN runtime.

        Raises:
            RuntimeError: If RKNN runtime initialization fails.
        """
        self._rknn = RKNN()

        ret = self._rknn.load_rknn(self._model_path)
        if ret != 0:
            msg = f"Failed to load RKNN model: {ret}"
            raise RuntimeError(msg)

        if self._target and self._device_id:
            logger.info(
                "Initializing on-device runtime: target=%s, device=%s",
                self._target,
                self._device_id,
            )
            ret = self._rknn.init_runtime(
                target=self._target, device_id=self._device_id
            )
        else:
            logger.info("Initializing simulated runtime (PC)")
            ret = self._rknn.init_runtime(target=None)

        if ret != 0:
            msg = f"Failed to init RKNN runtime: {ret}"
            raise RuntimeError(msg)

    def infer(self, inputs: list[np.ndarray]) -> tuple[list[np.ndarray], float]:
        """Run inference on the given inputs.

        Args:
            inputs: List of numpy arrays matching the model's input spec.

        Returns:
            Tuple of (output arrays, elapsed time in milliseconds).

        Raises:
            RuntimeError: If runtime is not initialized.
        """
        if self._rknn is None:
            msg = "RKNN runtime not initialized. Call init() first."
            raise RuntimeError(msg)

        start = time.perf_counter()
        outputs = self._rknn.inference(inputs=inputs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return outputs, elapsed_ms

    def release(self) -> None:
        """Release the RKNN runtime resources."""
        if self._rknn is not None:
            self._rknn.release()
            self._rknn = None


def run_audio_inference(
    model_path: str,
    audio_paths: list[str],
    device_id: str | None = None,
) -> dict[str, Any]:
    """Run audio CNN INT8 inference on a list of audio files.

    Args:
        model_path: Path to the .rknn audio model.
        audio_paths: List of audio file paths.
        device_id: ADB device ID, or None for simulated mode.

    Returns:
        Dict with keys: predictions (list[str]), confidences (list[float]),
        timings (list[float]).
    """
    target = "rk3576" if device_id else None
    runner = RKNNRunner(model_path, target=target, device_id=device_id)
    runner.init()

    predictions: list[str] = []
    confidences: list[float] = []
    timings: list[float] = []
    classes = ["eating", "drinking", "vomiting", "ambient", "other"]

    try:
        for audio_path in audio_paths:
            features = _load_audio_features(audio_path)
            outputs, elapsed_ms = runner.infer([features])

            logits = outputs[0].flatten()
            probs = _softmax(logits)
            pred_idx = int(np.argmax(probs))

            predictions.append(classes[pred_idx])
            confidences.append(float(probs[pred_idx]))
            timings.append(elapsed_ms)
    finally:
        runner.release()

    return {
        "predictions": predictions,
        "confidences": confidences,
        "timings": timings,
    }


def _load_audio_features(audio_path: str) -> np.ndarray:
    """Load audio file and extract log-mel spectrogram features.

    Args:
        audio_path: Path to an audio file.

    Returns:
        Numpy array of shape [1, 1, 64, T] (batch, channel, n_mels, time).
    """
    import torchaudio

    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_mels=64
    )
    mel = mel_transform(waveform)
    log_mel = (mel + 1e-8).log()

    # Shape: [1, 1, 64, T]
    return log_mel.unsqueeze(0).numpy()


def _softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax for a 1D array."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
