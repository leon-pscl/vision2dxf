"""Deterministic mock adapters.

Rules for everything in this package:
  * same input, same output, always (seeded randomness only)
  * is_mock=True on every result
  * no image data ever reaches disk
"""

from __future__ import annotations

import base64
import hashlib
import random
import struct
import zlib

from registry import load_config


def seeded(*parts: object) -> random.Random:
    """A Random seeded from the inputs, so mocks are reproducible."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return random.Random(int(digest[:16], 16))


def image_size(data_url_or_b64: str) -> tuple[int, int]:
    """Pixel size of a base64 PNG or JPEG. Falls back to a sane default."""
    raw = _decode(data_url_or_b64)
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        # IHDR width/height are the first two big-endian uint32s
        w, h = struct.unpack(">II", raw[16:24])
        return int(w), int(h)
    if raw[:2] == b"\xff\xd8":
        return _jpeg_size(raw)
    return 480, 640


def _decode(data_url_or_b64: str) -> bytes:
    s = data_url_or_b64
    if "," in s[:64] and s.lstrip().startswith("data:"):
        s = s.split(",", 1)[1]
    try:
        return base64.b64decode(s, validate=False)
    except Exception:
        return b""


def _jpeg_size(raw: bytes) -> tuple[int, int]:
    i = 2
    n = len(raw)
    while i < n - 9:
        if raw[i] != 0xFF:
            i += 1
            continue
        marker = raw[i + 1]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            h, w = struct.unpack(">HH", raw[i + 5 : i + 9])
            return int(w), int(h)
        seg = struct.unpack(">H", raw[i + 2 : i + 4])[0]
        i += 2 + seg
    return 480, 640


# --------------------------------------------------------------------------
# a real, valid PNG, built in memory so nothing touches disk
# --------------------------------------------------------------------------


def _png(width: int, height: int, pixels: bytearray) -> bytes:
    """Encode RGB rows into a PNG. ``pixels`` is width*height*3 bytes."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type 0
        raw.extend(pixels[y * width * 3 : (y + 1) * width * 3])

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


def mask_png(width: int, height: int, seed_key: str, cx_frac: float = 0.5) -> str:
    """A person-shaped binary mask as base64 PNG. White figure, black ground."""
    rnd = seeded(seed_key)
    # body runs down the centre; head is a circle above it
    cx = width * cx_frac
    head_r = width * 0.13
    head_cy = height * 0.11
    shoulder_y = height * 0.24
    hip_y = height * 0.52
    foot_y = height * 0.97

    shoulder_half = width * 0.20 + rnd.uniform(-3, 3)
    waist_half = width * 0.15
    hip_half = width * 0.185
    leg_half = width * 0.055

    px = bytearray(width * height * 3)
    for y in range(height):
        fy = y + 0.5
        if fy <= shoulder_y:
            dy = fy - head_cy
            half = head_r * (1 - dy * dy / (head_r * head_r)) ** 0.5 if abs(dy) < head_r else 0.0
            if abs(dy) < head_r * 0.75:  # neck
                half = width * 0.045
        elif fy <= hip_y:
            t = (fy - shoulder_y) / (hip_y - shoulder_y)
            half = shoulder_half + (hip_half - shoulder_half) * t
            half = max(waist_half * 0.9, half)
        else:
            t = min(1.0, (fy - hip_y) / (foot_y - hip_y))
            half = leg_half * (1.0 - 0.15 * t)

        if half <= 0:
            continue
        x0 = max(0, int(cx - half))
        x1 = min(width, int(cx + half) + 1)
        for x in range(x0, x1):
            i = (y * width + x) * 3
            px[i] = px[i + 1] = px[i + 2] = 255

    return base64.b64encode(_png(width, height, px)).decode("ascii")


def placeholder_png(width: int, height: int, label: str) -> str:
    """A flat card with a border, for the step 10 drape placeholder."""
    px = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            i = (y * width + x) * 3
            edge = x < 2 or y < 2 or x >= width - 2 or y >= height - 2
            v = 255 if edge else (246 if (x + y) % 8 else 244)
            px[i] = px[i + 1] = px[i + 2] = v
    return base64.b64encode(_png(width, height, px)).decode("ascii")


def cfg() -> dict:
    return load_config()
