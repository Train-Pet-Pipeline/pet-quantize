"""PET_ALLOW_MISSING_SDK env gate tests.

P2-C added real SDK-gated imports (rkllm cluster). Gate behavior now:
- PET_ALLOW_MISSING_SDK=1 → register_all() succeeds, warns about missing SDK
- PET_ALLOW_MISSING_SDK unset → register_all() raises ModuleNotFoundError
"""
from __future__ import annotations

import sys

import pytest


def test_missing_sdk_env_set_passes(monkeypatch) -> None:
    """register_all() succeeds when PET_ALLOW_MISSING_SDK=1 is set."""
    monkeypatch.setenv("PET_ALLOW_MISSING_SDK", "1")
    sys.modules.pop("pet_quantize.plugins._register", None)
    from pet_quantize.plugins._register import register_all
    register_all()


def test_missing_sdk_env_unset_raises_on_gated_import(monkeypatch) -> None:
    """register_all() raises ModuleNotFoundError when any gated SDK missing and gate unset.

    Whichever cluster (rknn or rkllm) fires first depends on install order in the
    local env; the test asserts gate behavior, not which SDK happened to trip it.
    """
    monkeypatch.delenv("PET_ALLOW_MISSING_SDK", raising=False)
    sys.modules.pop("pet_quantize.plugins._register", None)
    from pet_quantize.plugins._register import register_all
    with pytest.raises(ModuleNotFoundError, match=r"rknn|rkllm"):
        register_all()
