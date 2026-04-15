"""Build release package with tarball and manifest.json.

Collects converted model files and prompt files from pet-schema,
generates manifest with sha256 checksums, and creates a signed tarball.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Mapping of model file patterns to manifest keys
_MODEL_FILE_MAP = {
    "vision_*.rknn": "vision_encoder",
    "qwen2vl_*.rkllm": "llm",
    "audio_*.rknn": "audio",
}


def _sha256(data: bytes) -> str:
    """Compute SHA-256 hex digest of data."""
    return hashlib.sha256(data).hexdigest()


def _get_prompt_files(schema_version: str) -> dict[str, tuple[str, bytes]]:
    """Read prompt files from the installed pet-schema package.

    Args:
        schema_version: Schema version string (e.g., "1.0").

    Returns:
        Dict mapping manifest key to (filename, content bytes).
    """
    from pet_schema import get_prompt_paths

    paths = get_prompt_paths(schema_version)
    result = {}
    for key, path in paths.items():
        p = Path(path)
        result[key] = (p.name, p.read_bytes())
    return result


def _collect_model_files(converted_dir: str) -> dict[str, tuple[str, bytes]]:
    """Collect model files from the converted artifacts directory.

    Args:
        converted_dir: Path to artifacts/converted/.

    Returns:
        Dict mapping manifest key to (filename, content bytes).
    """
    import glob

    result = {}
    for pattern, key in _MODEL_FILE_MAP.items():
        matches = glob.glob(str(Path(converted_dir) / pattern))
        if not matches:
            logger.warning("No model file matching '%s' in %s", pattern, converted_dir)
            continue
        file_path = Path(matches[0])
        result[key] = (file_path.name, file_path.read_bytes())
    return result


def build_package(
    converted_dir: str,
    release_dir: str,
    config: dict[str, Any],
) -> dict[str, str]:
    """Build the release package.

    Args:
        converted_dir: Path to artifacts/converted/.
        release_dir: Path to artifacts/release/.
        config: Packaging section of params.yaml.

    Returns:
        Dict with keys: tarball_path, manifest_path.
    """
    version = config["version"]
    lora_version = config.get("lora_version", "")
    min_firmware = config.get("min_firmware", "2.0.0")
    release_notes = config.get("release_notes", "")

    Path(release_dir).mkdir(parents=True, exist_ok=True)

    # Collect all files
    model_files = _collect_model_files(converted_dir)

    try:
        from pet_schema import __version__ as schema_pkg_version
    except ImportError:
        schema_pkg_version = "1.0"

    schema_version = schema_pkg_version
    prompt_version = schema_pkg_version

    prompt_files = _get_prompt_files(schema_version)

    all_files = {**model_files, **prompt_files}

    # Build manifest
    files_manifest: dict[str, Any] = {}
    for key, (filename, content) in all_files.items():
        files_manifest[key] = {
            "path": filename,
            "sha256": _sha256(content),
            "size_bytes": len(content),
        }

    manifest = {
        "version": version,
        "schema_version": schema_version,
        "prompt_version": prompt_version,
        "lora_version": lora_version,
        "min_firmware": min_firmware,
        "build_timestamp": datetime.now(UTC).isoformat(),
        "files": files_manifest,
        "release_notes": release_notes,
    }

    manifest_path = str(Path(release_dir) / "manifest.json")
    Path(manifest_path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info("Manifest written to %s", manifest_path)

    # Create tarball
    tarball_name = f"pet-model-v{version}.tar.gz"
    tarball_path = str(Path(release_dir) / tarball_name)

    with tarfile.open(tarball_path, "w:gz") as tar:
        # Add manifest
        tar.add(manifest_path, arcname="manifest.json")

        # Add all files
        for _key, (filename, content) in all_files.items():
            info = tarfile.TarInfo(name=filename)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))

    logger.info("Tarball created: %s", tarball_path)

    return {"tarball_path": tarball_path, "manifest_path": manifest_path}
