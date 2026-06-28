"""Dataset fetch, verify, and extract helpers."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

from tcsr.utils.logging import get_logger

log = get_logger(__name__)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksum(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"Checksum mismatch for {path}\n  expected: {expected}\n  actual:   {actual}"
        )
    log.info("Checksum OK: %s", path.name)


def load_checksums(checksum_file: Path) -> dict[str, str]:
    """Parse a .sha256 file: '<hash>  <filename>' per line."""
    result = {}
    for line in checksum_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            result[parts[1].strip()] = parts[0].strip()
    return result


def verify_dataset(raw_dir: Path, checksum_file: Path) -> None:
    """Verify all files listed in a .sha256 manifest."""
    checksums = load_checksums(checksum_file)
    for filename, expected_hash in checksums.items():
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Expected dataset file missing: {path}")
        verify_checksum(path, expected_hash)
    log.info("All %d checksums verified for %s", len(checksums), raw_dir.name)
