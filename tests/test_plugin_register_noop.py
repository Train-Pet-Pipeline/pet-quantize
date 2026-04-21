"""NoopConverter + entry-point discovery registration tests."""
from __future__ import annotations

import pytest


def test_noop_registers_via_register_all(monkeypatch) -> None:
    """register_all() triggers @CONVERTERS.register_module side-effect."""
    import sys

    from pet_infra.registry import CONVERTERS

    # Clean slate: pop the registry entry and all related cached modules so that
    # register_all() re-imports noop.py and fires the decorator side-effect.
    # We must also pop the converters package itself; otherwise Python finds the
    # noop attribute on the cached package object and skips module re-execution.
    CONVERTERS._module_dict.pop("noop_converter", None)
    sys.modules.pop("pet_quantize.plugins.converters.noop", None)
    sys.modules.pop("pet_quantize.plugins.converters", None)
    sys.modules.pop("pet_quantize.plugins._register", None)

    from pet_quantize.plugins._register import register_all

    register_all()
    assert "noop_converter" in CONVERTERS.module_dict


def test_entry_point_discoverable() -> None:
    """pet_quantize entry-point is discoverable via importlib.metadata."""
    from importlib.metadata import entry_points

    eps = entry_points(group="pet_infra.plugins")
    assert "pet_quantize" in {ep.name for ep in eps}
