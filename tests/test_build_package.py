"""Tests for pet_quantize.packaging.build_package."""
from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture()
def converted_dir(tmp_dir: Path) -> Path:
    """Create a dummy converted artifacts directory."""
    d = tmp_dir / "converted"
    d.mkdir()
    (d / "vision_rk3576.rknn").write_bytes(b"vision_data" * 100)
    (d / "qwen2vl_2b_w8a8_rk3576.rkllm").write_bytes(b"llm_data" * 100)
    (d / "audio_cnn_int8.rknn").write_bytes(b"audio_data" * 100)
    return d


@pytest.fixture()
def release_dir(tmp_dir: Path) -> Path:
    """Create release output directory."""
    d = tmp_dir / "release"
    d.mkdir()
    return d


def test_build_creates_tarball_and_manifest(
    converted_dir: Path,
    release_dir: Path,
    sample_params: dict[str, Any],
) -> None:
    """build_package creates a tarball and manifest.json."""
    from pet_quantize.packaging.build_package import build_package

    config = sample_params["packaging"]
    config["version"] = "1.0.0"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "pet_quantize.packaging.build_package._get_prompt_files",
            lambda _: {
                "prompt_system": ("prompt_system_v1.0.txt", b"system prompt"),
                "prompt_user": ("prompt_user_v1.0.jinja2", b"user prompt"),
            },
        )
        result = build_package(
            converted_dir=str(converted_dir),
            release_dir=str(release_dir),
            config=config,
        )

    assert Path(result["tarball_path"]).exists()
    assert Path(result["manifest_path"]).exists()


def test_manifest_contains_all_files(
    converted_dir: Path,
    release_dir: Path,
    sample_params: dict[str, Any],
) -> None:
    """Manifest lists all model files with sha256 and size_bytes."""
    from pet_quantize.packaging.build_package import build_package

    config = sample_params["packaging"]
    config["version"] = "1.0.0"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "pet_quantize.packaging.build_package._get_prompt_files",
            lambda _: {
                "prompt_system": ("prompt_system_v1.0.txt", b"system prompt"),
                "prompt_user": ("prompt_user_v1.0.jinja2", b"user prompt"),
            },
        )
        result = build_package(
            converted_dir=str(converted_dir),
            release_dir=str(release_dir),
            config=config,
        )

    manifest = json.loads(Path(result["manifest_path"]).read_text())

    assert "vision_encoder" in manifest["files"]
    assert "llm" in manifest["files"]
    assert "audio" in manifest["files"]
    assert "prompt_system" in manifest["files"]
    assert "prompt_user" in manifest["files"]

    for file_info in manifest["files"].values():
        assert "sha256" in file_info
        assert "size_bytes" in file_info
        assert "path" in file_info
        assert len(file_info["sha256"]) == 64


def test_manifest_contains_metadata(
    converted_dir: Path,
    release_dir: Path,
    sample_params: dict[str, Any],
) -> None:
    """Manifest contains all required metadata fields."""
    from pet_quantize.packaging.build_package import build_package

    config = sample_params["packaging"]
    config["version"] = "1.2.0"
    config["lora_version"] = "1.2"
    config["release_notes"] = "Test release"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "pet_quantize.packaging.build_package._get_prompt_files",
            lambda _: {
                "prompt_system": ("prompt_system_v1.0.txt", b"system prompt"),
                "prompt_user": ("prompt_user_v1.0.jinja2", b"user prompt"),
            },
        )
        result = build_package(
            converted_dir=str(converted_dir),
            release_dir=str(release_dir),
            config=config,
        )

    manifest = json.loads(Path(result["manifest_path"]).read_text())

    assert manifest["version"] == "1.2.0"
    assert manifest["lora_version"] == "1.2"
    assert manifest["min_firmware"] == "2.0.0"
    assert manifest["release_notes"] == "Test release"
    assert "schema_version" in manifest
    assert "prompt_version" in manifest
    assert "build_timestamp" in manifest


def test_tarball_contains_all_files(
    converted_dir: Path,
    release_dir: Path,
    sample_params: dict[str, Any],
) -> None:
    """Tarball contains all model, prompt, and manifest files."""
    from pet_quantize.packaging.build_package import build_package

    config = sample_params["packaging"]
    config["version"] = "1.0.0"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "pet_quantize.packaging.build_package._get_prompt_files",
            lambda _: {
                "prompt_system": ("prompt_system_v1.0.txt", b"system prompt"),
                "prompt_user": ("prompt_user_v1.0.jinja2", b"user prompt"),
            },
        )
        result = build_package(
            converted_dir=str(converted_dir),
            release_dir=str(release_dir),
            config=config,
        )

    with tarfile.open(result["tarball_path"], "r:gz") as tar:
        names = tar.getnames()

    assert any("manifest.json" in n for n in names)
    assert any(".rknn" in n for n in names)
    assert any(".rkllm" in n for n in names)
    assert any("prompt_system" in n for n in names)
