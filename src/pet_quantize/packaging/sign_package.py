"""RSA-2048 package signing.

Signs the release tarball. In local development where no private key is
available, signing is skipped with a warning.
"""
from __future__ import annotations

import logging
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

logger = logging.getLogger(__name__)


def sign_package(
    tarball_path: str,
    private_key_path: str,
) -> str | None:
    """Sign a tarball with RSA-2048.

    Args:
        tarball_path: Path to the tarball file.
        private_key_path: Path to the RSA private key PEM file.
            Empty string or nonexistent path triggers skip.

    Returns:
        Path to the .sig signature file, or None if signing was skipped.

    Raises:
        FileNotFoundError: If tarball does not exist.
    """
    if not Path(tarball_path).exists():
        msg = f"Tarball not found: {tarball_path}"
        raise FileNotFoundError(msg)

    if not private_key_path or not Path(private_key_path).exists():
        logger.warning(
            "RSA private key not available, skipping signing. "
            "Set RSA_PRIVATE_KEY_PATH for production builds."
        )
        return None

    # Load private key
    key_data = Path(private_key_path).read_bytes()
    private_key = serialization.load_pem_private_key(key_data, password=None)

    # Read tarball
    tarball_data = Path(tarball_path).read_bytes()

    # Sign
    signature = private_key.sign(
        tarball_data,
        padding.PKCS1v15(),
        hashes.SHA256(),
    )

    # Write signature
    sig_path = tarball_path + ".sig"
    Path(sig_path).write_bytes(signature)

    logger.info("Package signed: %s", sig_path)
    return sig_path
