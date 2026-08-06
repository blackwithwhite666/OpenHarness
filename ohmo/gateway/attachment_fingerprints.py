"""Bounded, privacy-safe attachment fingerprints for inbound channel media.

The trusted gateway computes these descriptors; the model never authors them.
A descriptor contains only a SHA-256 digest, image dimensions, and a fixed,
EXIF-normalized DCT perceptual hash.  Paths, filenames, download tokens, and
image bytes are never included.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

ATTACHMENT_FINGERPRINT_MAX = 8
ATTACHMENT_FINGERPRINT_MAX_BYTES = 64 * 1024 * 1024
PHASH_ALGORITHM = "dct-phash-16x16-v1"
PHASH_HAMMING_THRESHOLD = 4

_PHASH_SIZE = 16
_DCT_INPUT_SIZE = 32
_MAX_DECODE_PIXELS = 40_000_000
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SOF_MARKERS = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)


def _dct_matrix(size: int) -> np.ndarray:
    positions = np.arange(size, dtype=np.float64)
    frequencies = np.arange(size, dtype=np.float64)[:, None]
    matrix = np.cos(np.pi * (2 * positions + 1) * frequencies / (2 * size))
    matrix[0] *= 1 / np.sqrt(2)
    return matrix * np.sqrt(2 / size)


_DCT = _dct_matrix(_DCT_INPUT_SIZE)


def fingerprint_image_bytes(data: bytes) -> dict[str, object] | None:
    """Return a bounded descriptor for image bytes, or None for non-images."""
    dimensions = _image_dimensions(data)
    if dimensions is None:
        return None
    width, height = dimensions
    descriptor: dict[str, object] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "width": width,
        "height": height,
    }
    phash = _dct_phash(data)
    if phash is not None:
        descriptor["phash"] = phash
        descriptor["phash_algorithm"] = PHASH_ALGORITHM
    return descriptor


def fingerprint_image_file(media_path: str | os.PathLike[str]) -> dict[str, object] | None:
    """Fingerprint one downloaded image; never leaks the path into the result."""
    try:
        path = Path(media_path)
        if not path.is_file() or path.stat().st_size > ATTACHMENT_FINGERPRINT_MAX_BYTES:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    return fingerprint_image_bytes(data)


def compute_attachment_fingerprints(media_paths: list[str] | None) -> list[dict[str, object]]:
    """Fingerprint inbound image attachments, bounded to ATTACHMENT_FINGERPRINT_MAX."""
    fingerprints: list[dict[str, object]] = []
    for media_path in media_paths or []:
        if len(fingerprints) >= ATTACHMENT_FINGERPRINT_MAX:
            break
        try:
            descriptor = fingerprint_image_file(media_path)
        except Exception:
            logger.exception("ohmo attachment fingerprint failed")
            continue
        if descriptor is not None:
            fingerprints.append(descriptor)
    return fingerprints


def phash_hamming_distance(left: str, right: str) -> int | None:
    """Return the Hamming distance for two fixed-version hexadecimal pHashes."""
    if len(left) != len(right) or not left or not right:
        return None
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError:
        return None


def _image_dimensions(data: bytes) -> tuple[int, int] | None:
    """Parse (width, height) from PNG/JPEG/GIF/BMP headers; None when unknown."""
    if data.startswith(_PNG_SIGNATURE) and len(data) >= 24 and data[12:16] == b"IHDR":
        width, height = struct.unpack(">II", data[16:24])
        return (width, height) if width > 0 and height > 0 else None
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        width, height = struct.unpack("<HH", data[6:10])
        return (width, height) if width > 0 and height > 0 else None
    if data[:2] == b"BM" and len(data) >= 26:
        width, height = struct.unpack("<ii", data[18:26])
        return (width, abs(height)) if width > 0 and height != 0 else None
    if data[:2] == b"\xff\xd8":
        return _jpeg_dimensions(data)
    return None


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    pos = 2
    while pos + 4 <= len(data):
        if data[pos] != 0xFF:
            pos += 1
            continue
        marker = data[pos + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        segment_length = struct.unpack(">H", data[pos + 2 : pos + 4])[0]
        if segment_length < 2 or pos + 2 + segment_length > len(data):
            return None
        if marker in _JPEG_SOF_MARKERS:
            if segment_length < 7:
                return None
            height, width = struct.unpack(">HH", data[pos + 5 : pos + 9])
            return (width, height) if width > 0 and height > 0 else None
        pos += 2 + segment_length
    return None


def _dct_phash(data: bytes) -> str | None:
    try:
        with Image.open(io.BytesIO(data)) as source:
            image = ImageOps.exif_transpose(source)
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > _MAX_DECODE_PIXELS:
                return None
            grayscale = np.asarray(
                image.convert("L").resize(
                    (_DCT_INPUT_SIZE, _DCT_INPUT_SIZE), Image.Resampling.BILINEAR
                ),
                dtype=np.float64,
            )
    except Exception:  # noqa: BLE001 - any decode failure just drops the pHash
        return None

    coefficients = _DCT @ grayscale @ _DCT.T
    low_frequency = coefficients[:_PHASH_SIZE, :_PHASH_SIZE].copy()
    low_frequency[0, 0] = 0
    median = float(np.median(low_frequency))
    bits = 0
    for coefficient in low_frequency.reshape(-1):
        bits = (bits << 1) | int(coefficient >= median)
    return f"{bits:0{_PHASH_SIZE * _PHASH_SIZE // 4}x}"
