"""Tests for pet_quantize.packaging.verify_package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


@pytest.fixture()
def rsa_key_pair(tmp_dir: Path) -> tuple[Path, Path]:
    """Generate a test RSA key pair."""
    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048
    )
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    priv_path = tmp_dir / "test_private.pem"
    pub_path = tmp_dir / "test_public.pem"
    priv_path.write_bytes(private_pem)
    pub_path.write_bytes(public_pem)
    return priv_path, pub_path


@pytest.fixture()
def signed_release(tmp_dir: Path, rsa_key_pair: tuple[Path, Path]) -> Path:
    """Create a dummy signed release directory."""
    release = tmp_dir / "release"
    release.mkdir()

    # Create model file
    model_data = b"model_content" * 100
    model_path = release / "vision_rk3576.rknn"
    model_path.write_bytes(model_data)

    # Create manifest
    manifest = {
        "version": "1.0.0",
        "files": {
            "vision_encoder": {
                "path": "vision_rk3576.rknn",
                "sha256": hashlib.sha256(model_data).hexdigest(),
                "size_bytes": len(model_data),
            }
        },
    }
    manifest_path = release / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    # Create tarball (just the model for simplicity)
    import tarfile
    tarball_path = release / "pet-model-v1.0.0.tar.gz"
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(manifest_path, arcname="manifest.json")
        tar.add(model_path, arcname="vision_rk3576.rknn")

    # Sign
    priv_path, _ = rsa_key_pair
    private_key = serialization.load_pem_private_key(
        priv_path.read_bytes(), password=None
    )
    signature = private_key.sign(
        tarball_path.read_bytes(),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )
    sig_path = release / "pet-model-v1.0.0.tar.gz.sig"
    sig_path.write_bytes(signature)

    return release


def test_verify_valid_package(
    signed_release: Path, rsa_key_pair: tuple[Path, Path]
) -> None:
    """A valid signed package passes verification."""
    from pet_quantize.packaging.verify_package import verify_package

    _, pub_path = rsa_key_pair
    result = verify_package(
        release_dir=str(signed_release),
        public_key_path=str(pub_path),
    )
    assert result.passed is True
    assert len(result.errors) == 0


def test_verify_tampered_sha256(
    signed_release: Path, rsa_key_pair: tuple[Path, Path]
) -> None:
    """Tampering with a file causes sha256 verification failure."""
    from pet_quantize.packaging.verify_package import verify_package

    # Tamper with the model file
    (signed_release / "vision_rk3576.rknn").write_bytes(b"tampered!")

    _, pub_path = rsa_key_pair
    result = verify_package(
        release_dir=str(signed_release),
        public_key_path=str(pub_path),
    )
    assert result.passed is False
    assert any("sha256" in e.lower() for e in result.errors)


def test_verify_no_signature_warns(signed_release: Path) -> None:
    """Missing signature gives warning but does not fail."""
    from pet_quantize.packaging.verify_package import verify_package

    # Remove the signature file
    for sig in signed_release.glob("*.sig"):
        sig.unlink()

    result = verify_package(
        release_dir=str(signed_release),
        public_key_path="",
    )
    assert result.passed is True
    assert len(result.warnings) > 0
    assert any("signature" in w.lower() for w in result.warnings)
