"""Parity test: __version__ in __init__.py must match importlib.metadata version."""
from __future__ import annotations

import importlib.metadata

import pet_quantize


def test_version_attribute_matches_metadata() -> None:
    """pet_quantize.__version__ must equal the installed package version from pip metadata."""
    installed = importlib.metadata.version("pet-quantize")
    assert pet_quantize.__version__ == installed, (
        f"pet_quantize.__version__ ({pet_quantize.__version__!r}) does not match "
        f"installed package metadata ({installed!r}). "
        "Update src/pet_quantize/__init__.py to match pyproject.toml version."
    )
