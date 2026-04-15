"""Pytest fixtures for smoke validation with dual-mode support.

Provides device_id, model_dir, and params fixtures. Device mode is
activated via --device-id CLI option.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --device-id option for on-device testing."""
    parser.addoption(
        "--device-id",
        action="store",
        default=None,
        help="ADB device serial number for on-device testing",
    )


@pytest.fixture()
def device_id(request: pytest.FixtureRequest) -> str | None:
    """Get the device ID from CLI option."""
    return request.config.getoption("--device-id")


@pytest.fixture()
def params() -> dict:
    """Load params.yaml from the project root."""
    params_path = Path(__file__).resolve().parents[3] / "params.yaml"
    with open(params_path) as fh:
        return yaml.safe_load(fh)


@pytest.fixture()
def model_dir(params: dict) -> str:
    """Get the converted model directory path."""
    return params["convert"]["output_dir"] or "artifacts/converted"
