"""Bounded, privacy-safe attachment fingerprints for inbound channel media.

The trusted gateway computes these descriptors; the model never authors them.
A descriptor contains only a SHA-256 digest, image dimensions, and (when
decoding succeeds) a fixed-version perceptual hash. Paths, filenames, download
tokens, and image bytes are never included.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import struct
import zlib
from pathlib import Path

logger = logging.getLogger(__name__)

ATTACHMENT_FINGERPRINT_MAX = 8
ATTACHMENT_FINGERPRINT_MAX_BYTES = 64 * 1024 * 1024
PHASH_ALGORITHM = "ahash-16x16-gray-v1"

_PHASH_SIZE = 16
_MAX_DECODE_PIXELS = 40_000_000
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SOF_MARKERS = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)


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
    gray = _decode_grayscale(data, width, height)
    if gray is not None:
        phash = _ahash(gray, width, height)
        if phash is not None:
            descriptor["phash"] = phash
            descriptor["phash_algorithm"] = PHASH_ALGORITHM
    return descriptor


def fingerprint_image_file(media_path: str | os.PathLike[str]) -> dict[str, object] | None:
    """Fingerprint one downloaded image; never leaks the path into the result."""
    try:
        path = Path(media_path)
        if not path.is_file():
            return None
        if path.stat().st_size > ATTACHMENT_FINGERPRINT_MAX_BYTES:
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
        except Exception:  # noqa: BLE001 — fingerprinting is best-effort provenance
            logger.exception("ohmo attachment fingerprint failed")
            continue
        if descriptor is not None:
            fingerprints.append(descriptor)
    return fingerprints


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


def _decode_grayscale(data: bytes, width: int, height: int) -> list[int] | None:
    """Decode to 8-bit grayscale samples, or None when decoding fails."""
    if width <= 0 or height <= 0 or width * height > _MAX_DECODE_PIXELS:
        return None
    decoded = _decode_grayscale_pil(data)
    if decoded is not None:
        pixels, decoded_width, decoded_height = decoded
        if decoded_width == width and decoded_height == height:
            return pixels
        return None
    if data.startswith(_PNG_SIGNATURE):
        return _decode_png_grayscale(data, width, height)
    return None


def _decode_grayscale_pil(data: bytes) -> tuple[list[int], int, int] | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(io.BytesIO(data)) as image:
            grayscale = image.convert("L")
            width, height = grayscale.size
            if width * height > _MAX_DECODE_PIXELS:
                return None
            return list(grayscale.tobytes()), width, height
    except Exception:  # noqa: BLE001 — any decode failure just drops the phash
        return None


def _decode_png_grayscale(data: bytes, width: int, height: int) -> list[int] | None:
    """Stdlib decoder for 8-bit non-interlaced PNG (gray/RGB/gray+alpha/RGBA)."""
    pos = len(_PNG_SIGNATURE)
    bit_depth = color_type = interlace = None
    idat = bytearray()
    while pos + 8 <= len(data):
        chunk_length = struct.unpack(">I", data[pos : pos + 4])[0]
        chunk_type = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + chunk_length]
        if len(chunk) != chunk_length:
            return None
        if chunk_type == b"IHDR":
            if chunk_length != 13:
                return None
            (
                ihdr_width,
                ihdr_height,
                bit_depth,
                color_type,
                _compression,
                _filter_method,
                interlace,
            ) = struct.unpack(">IIBBBBB", chunk)
            if (ihdr_width, ihdr_height) != (width, height):
                return None
        elif chunk_type == b"IDAT":
            idat += chunk
        elif chunk_type == b"IEND":
            break
        pos += 12 + chunk_length
    channels_by_type = {0: 1, 2: 3, 4: 2, 6: 4}
    if bit_depth != 8 or interlace != 0 or color_type not in channels_by_type or not idat:
        return None
    channels = channels_by_type[color_type]
    try:
        raw = zlib.decompress(bytes(idat))
    except zlib.error:
        return None
    stride = width * channels
    if len(raw) != (stride + 1) * height:
        return None
    grayscale: list[int] = []
    previous = bytearray(stride)
    offset = 0
    for _row in range(height):
        filter_type = raw[offset]
        offset += 1
        line = bytearray(raw[offset : offset + stride])
        offset += stride
        if filter_type == 1:
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif filter_type == 2:
            for i in range(stride):
                line[i] = (line[i] + previous[i]) & 0xFF
        elif filter_type == 3:
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((left + previous[i]) >> 1)) & 0xFF
        elif filter_type == 4:
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                up = previous[i]
                up_left = previous[i - channels] if i >= channels else 0
                line[i] = (line[i] + _paeth(left, up, up_left)) & 0xFF
        elif filter_type != 0:
            return None
        previous = line
        if color_type in (0, 4):
            grayscale.extend(line[::channels])
        else:
            for i in range(0, stride, channels):
                grayscale.append((line[i] * 299 + line[i + 1] * 587 + line[i + 2] * 114) // 1000)
    return grayscale


def _paeth(left: int, up: int, up_left: int) -> int:
    estimate = left + up - up_left
    dist_left = abs(estimate - left)
    dist_up = abs(estimate - up)
    dist_up_left = abs(estimate - up_left)
    if dist_left <= dist_up and dist_left <= dist_up_left:
        return left
    if dist_up <= dist_up_left:
        return up
    return up_left


def _ahash(grayscale: list[int], width: int, height: int) -> str | None:
    """16x16 average hash (PHASH_ALGORITHM); deterministic box downsample."""
    if len(grayscale) != width * height:
        return None
    cells: list[float] = []
    for cell_y in range(_PHASH_SIZE):
        y0 = cell_y * height // _PHASH_SIZE
        y1 = max((cell_y + 1) * height // _PHASH_SIZE, y0 + 1)
        for cell_x in range(_PHASH_SIZE):
            x0 = cell_x * width // _PHASH_SIZE
            x1 = max((cell_x + 1) * width // _PHASH_SIZE, x0 + 1)
            total = 0
            count = 0
            for y in range(y0, min(y1, height)):
                row_base = y * width
                for x in range(x0, min(x1, width)):
                    total += grayscale[row_base + x]
                    count += 1
            cells.append(total / count)
    mean = sum(cells) / len(cells)
    bits = 0
    for cell in cells:
        bits = (bits << 1) | (1 if cell >= mean else 0)
    return f"{bits:0{_PHASH_SIZE * _PHASH_SIZE // 4}x}"
