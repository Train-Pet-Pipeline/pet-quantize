"""Package verification: sha256 integrity and RSA signature check.

Verifies the release package integrity. Missing signature triggers a
warning (not failure) for local development compatibility.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerifyResult:
    """Result of package verification."""

    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def verify_package(
    release_dir: str,
    public_key_path: str = "",
) -> VerifyResult:
    """Verify a release package's integrity and signature.

    Args:
        release_dir: Path to artifacts/release/.
        public_key_path: Path to RSA public key PEM. Empty to skip signature check.

    Returns:
        VerifyResult with pass/fail, errors, and warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []
    release = Path(release_dir)

    # Load manifest
    manifest_path = release / "manifest.json"
    if not manifest_path.exists():
        return VerifyResult(passed=False, errors=["manifest.json not found"])

    manifest = json.loads(manifest_path.read_text())

    # Verify sha256 of each file
    for key, file_info in manifest.get("files", {}).items():
        file_path = release / file_info["path"]
        if not file_path.exists():
            errors.append(f"{key}: file not found at {file_info['path']}")
            continue

        actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        expected_hash = file_info["sha256"]
        if actual_hash != expected_hash:
            errors.append(
                f"{key}: SHA256 mismatch — expected={expected_hash[:16]}..., "
                f"actual={actual_hash[:16]}..."
            )

    # Verify signature
    tarball_files = list(release.glob("*.tar.gz"))
    sig_files = list(release.glob("*.sig"))

    if not sig_files:
        warnings.append("No signature file found. Package is unsigned.")
    elif not public_key_path or not Path(public_key_path).exists():
        warnings.append(
            "Signature file present but no public key provided. "
            "Cannot verify signature."
        )
    else:
        tarball_path = tarball_files[0]
        sig_path = sig_files[0]

        pub_key_data = Path(public_key_path).read_bytes()
        public_key = serialization.load_pem_public_key(pub_key_data)
        assert isinstance(public_key, RSAPublicKey)

        try:
            public_key.verify(
                sig_path.read_bytes(),
                tarball_path.read_bytes(),
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
            logger.info("RSA signature verified successfully")
        except Exception:
            errors.append("RSA signature verification failed")

    passed = len(errors) == 0

    if passed:
        logger.info("Package verification passed")
    else:
        for e in errors:
            logger.error("Verification error: %s", e)
    for w in warnings:
        logger.warning("Verification warning: %s", w)

    return VerifyResult(passed=passed, errors=errors, warnings=warnings)
