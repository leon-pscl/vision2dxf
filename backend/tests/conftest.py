import base64
import os
import struct
import sys
import zlib

import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


def _png(w: int, h: int, rgb: tuple[int, int, int]) -> bytes:
    px = bytes(rgb) * (w * h)

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    raw = b"".join(b"\x00" + px[y * w * 3 : (y + 1) * w * 3] for y in range(h))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


@pytest.fixture(scope="session")
def front_png_b64() -> str:
    return base64.b64encode(_png(200, 320, (210, 190, 175))).decode("ascii")


@pytest.fixture(scope="session")
def side_png_b64() -> str:
    return base64.b64encode(_png(160, 320, (200, 185, 170))).decode("ascii")


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    import main

    return TestClient(main.app)


@pytest.fixture
def session_id(client) -> str:
    return client.post("/api/session").json()["session_id"]


def run(client, sid: int | str, step: int, backend: str | None = None, **payload) -> dict:
    """POST one step. Keyword args become the step payload; backend is step 6 only."""
    body: dict = {"payload": payload}
    if backend:
        body["backend"] = backend
    return client.post(f"/api/session/{sid}/step/{step}", json=body).json()


def full_run(client, sid: str, garment: str, front: str, side: str) -> None:
    """Walk steps 1 through 10 once, asserting each response is 200."""
    r = run(client, sid, 1, garment_type=garment)
    assert r.get("is_mock") is True, r
    r = run(client, sid, 2, front_image=front, side_image=side, height_cm=178, weight_kg=74)
    assert "distance_ok" in r, r
    r = run(client, sid, 3, image=front)
    assert r.get("mask_png"), r
    r = run(client, sid, 4)
    assert r.get("px_per_cm"), r
    r = run(client, sid, 5, confirm=True)
    assert r.get("confirmed_by_user") is True, r
    r = run(client, sid, 6)
    assert r.get("measurements"), r
    r = run(client, sid, 7)
    assert "yaml" in r, r
    r = run(client, sid, 8)
    assert r.get("pieces"), r
    r = run(client, sid, 9)
    assert r.get("pieces"), r
    r = run(client, sid, 10)
    assert r.get("toile_checklist"), r
