"""PET_ALLOW_MISSING_SDK env gate tests.

Note: in P2-B the SDK-gated clusters are empty (no imports to fail), so the
gate's re-raise behavior cannot be exercised here yet. That test lands in
P2-C when real SDK-backed plugins register. These tests verify the affordance
exists and register_all() is callable under both env settings.
"""
from __future__ import annotations

import sys


def test_missing_sdk_env_set_passes(monkeypatch) -> None:
    """register_all() succeeds when PET_ALLOW_MISSING_SDK=1 is set."""
    monkeypatch.setenv("PET_ALLOW_MISSING_SDK", "1")
    sys.modules.pop("pet_quantize.plugins._register", None)
    from pet_quantize.plugins._register import register_all
    register_all()


def test_missing_sdk_env_unset_still_passes_when_gates_empty(monkeypatch) -> None:
    """register_all() succeeds when gates have no imports (P2-B state)."""
    monkeypatch.delenv("PET_ALLOW_MISSING_SDK", raising=False)
    sys.modules.pop("pet_quantize.plugins._register", None)
    from pet_quantize.plugins._register import register_all
    register_all()  # empty try-bodies don't raise ImportError
