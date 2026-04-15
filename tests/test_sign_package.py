"""Tests for pet_quantize.packaging.sign_package."""
from __future__ import annotations

from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


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
def tarball(tmp_dir: Path) -> Path:
    """Create a dummy tarball for signing."""
    p = tmp_dir / "release" / "pet-model-v1.0.0.tar.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"fake_tarball_content" * 100)
    return p


def test_sign_with_key_creates_sig_file(
    tarball: Path, rsa_key_pair: tuple[Path, Path]
) -> None:
    """Signing with a private key creates a .sig file."""
    from pet_quantize.packaging.sign_package import sign_package

    priv_path, _ = rsa_key_pair
    sig_path = sign_package(str(tarball), str(priv_path))

    assert sig_path is not None
    assert Path(sig_path).exists()
    assert sig_path.endswith(".sig")


def test_sign_without_key_returns_none(tarball: Path) -> None:
    """Signing without a private key skips and returns None."""
    from pet_quantize.packaging.sign_package import sign_package

    result = sign_package(str(tarball), "")
    assert result is None


def test_sign_nonexistent_key_returns_none(tarball: Path) -> None:
    """Signing with nonexistent key path skips and returns None."""
    from pet_quantize.packaging.sign_package import sign_package

    result = sign_package(str(tarball), "/nonexistent/key.pem")
    assert result is None


def test_sign_nonexistent_tarball_raises(
    rsa_key_pair: tuple[Path, Path],
) -> None:
    """Signing a nonexistent tarball raises FileNotFoundError."""
    from pet_quantize.packaging.sign_package import sign_package

    priv_path, _ = rsa_key_pair
    with pytest.raises(FileNotFoundError):
        sign_package("/nonexistent/tarball.tar.gz", str(priv_path))
