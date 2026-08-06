from __future__ import annotations

import hashlib
import struct
import zlib
from io import BytesIO

from PIL import Image, ImageDraw

from ohmo.gateway.attachment_fingerprints import (
    ATTACHMENT_FINGERPRINT_MAX,
    PHASH_ALGORITHM,
    PHASH_HAMMING_THRESHOLD,
    compute_attachment_fingerprints,
    fingerprint_image_bytes,
    fingerprint_image_file,
    phash_hamming_distance,
)


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    )


def _make_rgb_png(width: int, height: int, pixel) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    if callable(pixel):
        raw = b"".join(
            b"\x00" + b"".join(bytes(pixel(x, y)) for x in range(width)) for y in range(height)
        )
    else:
        row = bytes(pixel) * width
        raw = b"".join(b"\x00" + row for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )


def _make_jpeg_header(width: int, height: int) -> bytes:
    return (
        b"\xff\xd8"
        + b"\xff\xe0"
        + struct.pack(">H", 4)
        + b"\x00\x00"
        + b"\xff\xc0"
        + struct.pack(">H", 8 + 3)
        + b"\x08"
        + struct.pack(">HH", height, width)
        + b"\x03\x01\x11\x00"
        + b"\xff\xd9"
    )


def _make_jpeg(*, quality: int = 95, orientation: int | None = None) -> bytes:
    image = Image.new("RGB", (48, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 8, 30, 35), fill="red")
    draw.ellipse((16, 32, 44, 60), fill="blue")
    exif = image.getexif()
    if orientation is not None:
        exif[274] = orientation
    output = BytesIO()
    image.save(output, format="JPEG", quality=quality, exif=exif.tobytes())
    return output.getvalue()


def test_fingerprint_image_bytes_sha256_dimensions_and_phash() -> None:
    data = _make_rgb_png(32, 24, (200, 30, 30))

    descriptor = fingerprint_image_bytes(data)

    assert descriptor is not None
    assert descriptor["sha256"] == hashlib.sha256(data).hexdigest()
    assert descriptor["width"] == 32
    assert descriptor["height"] == 24
    assert descriptor["phash_algorithm"] == PHASH_ALGORITHM
    assert isinstance(descriptor["phash"], str)
    assert len(descriptor["phash"]) == 64
    int(descriptor["phash"], 16)


def test_fingerprint_phash_is_deterministic_for_identical_bytes() -> None:
    first = fingerprint_image_bytes(_make_rgb_png(20, 20, (10, 200, 30)))
    second = fingerprint_image_bytes(_make_rgb_png(20, 20, (10, 200, 30)))

    assert first == second


def test_fingerprint_phash_distinguishes_different_images() -> None:
    left_dark = fingerprint_image_bytes(
        _make_rgb_png(32, 32, lambda x, y: (0, 0, 0) if x < 16 else (255, 255, 255))
    )
    top_dark = fingerprint_image_bytes(
        _make_rgb_png(32, 32, lambda x, y: (0, 0, 0) if y < 16 else (255, 255, 255))
    )

    assert left_dark["sha256"] != top_dark["sha256"]
    assert left_dark["phash"] != top_dark["phash"]


def test_fingerprint_jpeg_dimensions_without_decode() -> None:
    data = _make_jpeg_header(640, 480)

    descriptor = fingerprint_image_bytes(data)

    assert descriptor is not None
    assert descriptor["sha256"] == hashlib.sha256(data).hexdigest()
    assert (descriptor["width"], descriptor["height"]) == (640, 480)


def test_fingerprint_valid_jpeg_exif_orientation_and_reencode_are_compatible() -> None:
    original = fingerprint_image_bytes(_make_jpeg())
    reencoded = fingerprint_image_bytes(_make_jpeg(quality=72))
    oriented_image = Image.open(BytesIO(_make_jpeg())).transpose(Image.Transpose.ROTATE_90)
    oriented_output = BytesIO()
    exif = oriented_image.getexif()
    exif[274] = 6
    oriented_image.save(oriented_output, format="JPEG", quality=95, exif=exif.tobytes())
    oriented = fingerprint_image_bytes(oriented_output.getvalue())

    assert original is not None and reencoded is not None and oriented is not None
    assert original["phash_algorithm"] == PHASH_ALGORITHM
    assert phash_hamming_distance(original["phash"], reencoded["phash"]) <= PHASH_HAMMING_THRESHOLD
    assert phash_hamming_distance(original["phash"], oriented["phash"]) <= PHASH_HAMMING_THRESHOLD


def test_phash_hamming_distance_rejects_malformed_or_different_width_values() -> None:
    assert phash_hamming_distance("0", "1") == 1
    assert phash_hamming_distance("0", "00") is None
    assert phash_hamming_distance("x", "0") is None


def test_fingerprint_rejects_non_image_bytes() -> None:
    assert fingerprint_image_bytes(b"not an image at all") is None
    assert fingerprint_image_bytes(b"") is None


def test_fingerprint_descriptor_excludes_paths_tokens_and_bytes(tmp_path) -> None:
    secret_path = tmp_path / "telegram_download_token_abc123.jpg"
    secret_path.write_bytes(_make_rgb_png(16, 16, (1, 2, 3)))

    descriptor = fingerprint_image_file(str(secret_path))

    assert descriptor is not None
    allowed_keys = {"sha256", "width", "height", "phash", "phash_algorithm"}
    assert set(descriptor.keys()) <= allowed_keys
    serialized = repr(descriptor)
    assert str(tmp_path) not in serialized
    assert "telegram_download_token_abc123" not in serialized
    assert str(secret_path) not in serialized


def test_fingerprint_file_missing_is_none(tmp_path) -> None:
    assert fingerprint_image_file(str(tmp_path / "missing.jpg")) is None


def test_compute_attachment_fingerprints_skips_non_images_and_missing(tmp_path) -> None:
    image = tmp_path / "a.jpg"
    image.write_bytes(_make_rgb_png(8, 8, (9, 9, 9)))
    text = tmp_path / "notes.txt"
    text.write_text("hello", encoding="utf-8")

    fingerprints = compute_attachment_fingerprints(
        [str(image), str(text), str(tmp_path / "missing.png")]
    )

    assert len(fingerprints) == 1
    assert fingerprints[0]["sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()


def test_compute_attachment_fingerprints_is_bounded(tmp_path) -> None:
    paths = []
    for index in range(ATTACHMENT_FINGERPRINT_MAX + 4):
        image = tmp_path / f"img{index}.png"
        image.write_bytes(_make_rgb_png(8, 8, (index, index, index)))
        paths.append(str(image))

    fingerprints = compute_attachment_fingerprints(paths)

    assert len(fingerprints) == ATTACHMENT_FINGERPRINT_MAX
