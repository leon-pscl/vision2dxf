"""End-to-end API behaviour, including the promise that no image hits disk."""

from __future__ import annotations

import os

import pytest

from conftest import full_run, run

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_create_and_fetch_session(client):
    sid = client.post("/api/session").json()["session_id"]
    state = client.get(f"/api/session/{sid}").json()
    assert state["session_id"] == sid
    assert state["results"] == {}
    assert state["any_mock"] is False


def test_unknown_session_404(client):
    assert client.get("/api/session/nope").status_code == 404


def test_cors_is_an_allowlist_not_a_wildcard(client):
    """No auth on any endpoint, so a wildcard origin would let any page the
    user visits drive a locally running instance."""
    r = client.options(
        "/api/session",
        headers={
            "Origin": "http://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}

    ok = client.options(
        "/api/session",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_sessions_are_evicted_past_the_cap(client):
    """Sessions hold base64 photos in memory; the cap stops unbounded growth."""
    import main

    cap = main.MAX_SESSIONS
    made = [client.post("/api/session").json()["session_id"] for _ in range(cap + 5)]
    assert len(main.SESSIONS) <= cap
    # the newest survive, the oldest are gone
    assert made[-1] in main.SESSIONS
    assert client.get(f"/api/session/{made[-1]}").status_code == 200
    assert client.get(f"/api/session/{made[0]}").status_code == 404
    for sid in made:
        assert sid not in main.INPUTS or len(main.SESSIONS) <= cap


def test_models_endpoint(client):
    body = client.get("/api/models").json()
    assert body["device"] in ("cpu", "cuda")
    assert body["adapters"]["segmentation"].startswith("adapters.")


@pytest.mark.parametrize("garment", ["polo_shirt", "long_sleeve_polo", "short_sleeve_polo", "slacks"])
def test_click_through_all_ten_steps(client, front_png_b64, side_png_b64, garment):
    sid = client.post("/api/session").json()["session_id"]
    full_run(client, sid, garment, front_png_b64, side_png_b64)

    state = client.get(f"/api/session/{sid}").json()
    for step in range(1, 11):
        assert str(step) in state["results"], f"step {step} missing"
        assert state["status"][str(step)]["status"] in ("done", "warning")
    assert state["any_mock"] is True, "the banner must show: everything is mock"


def test_step6_locked_until_landmarks_confirmed(client, front_png_b64, side_png_b64):
    sid = client.post("/api/session").json()["session_id"]
    run(client, sid, 1, garment_type="polo_shirt")
    run(client, sid, 2, front_image=front_png_b64, side_image=side_png_b64, height_cm=178)

    r = client.post(f"/api/session/{sid}/step/6", json={})
    assert r.status_code == 400
    assert "confirm" in r.json()["detail"].lower()

    run(client, sid, 5)
    assert client.post(f"/api/session/{sid}/step/6", json={}).status_code == 400

    run(client, sid, 5, confirm=True)
    assert client.post(f"/api/session/{sid}/step/6", json={}).status_code == 200


def test_steps_7_to_10_also_locked(client, front_png_b64, side_png_b64):
    sid = client.post("/api/session").json()["session_id"]
    run(client, sid, 1, garment_type="slacks")
    run(client, sid, 2, front_image=front_png_b64, side_image=side_png_b64, height_cm=178)
    for step in (7, 8, 9, 10):
        r = client.post(f"/api/session/{sid}/step/{step}", json={})
        assert r.status_code == 400, step


def test_dragging_a_landmark_marks_it_user(client, front_png_b64, side_png_b64):
    sid = client.post("/api/session").json()["session_id"]
    run(client, sid, 1, garment_type="polo_shirt")
    run(client, sid, 2, front_image=front_png_b64, side_image=side_png_b64, height_cm=178)
    base = run(client, sid, 5)
    assert base["points"]["waist"]["source"] == "heuristic"

    dragged = run(
        client, sid, 5, updates=[{"name": "waist", "x": 101.0, "y": 133.0}]
    )
    assert dragged["points"]["waist"]["source"] == "user"
    assert dragged["points"]["waist"]["x"] == 101.0
    assert dragged["points"]["waist"]["confidence"] == 1.0
    # untouched points keep their source
    assert dragged["points"]["hip"]["source"] == base["points"]["hip"]["source"]


def test_both_step6_backends_return_the_same_type(client, front_png_b64, side_png_b64):
    sid = client.post("/api/session").json()["session_id"]
    run(client, sid, 1, garment_type="polo_shirt")
    run(client, sid, 2, front_image=front_png_b64, side_image=side_png_b64, height_cm=178, weight_kg=74)
    run(client, sid, 5, confirm=True)

    reg = client.post(f"/api/session/{sid}/step/6", json={"backend": "regression"}).json()
    bm = client.post(f"/api/session/{sid}/step/6", json={"backend": "body_model"}).json()
    assert reg["backend"] == "regression"
    assert bm["backend"] == "body_model"
    assert set(reg["measurements"]) == set(bm["measurements"])
    for name in reg["measurements"]:
        for field in ("value_cm", "lower_cm", "upper_cm", "method"):
            assert type(reg["measurements"][name][field]) is type(bm["measurements"][name][field])


def test_step6_reports_only_the_measurements_the_garment_needs(
    client, front_png_b64, side_png_b64
):
    sid = client.post("/api/session").json()["session_id"]
    run(client, sid, 1, garment_type="slacks")
    run(client, sid, 2, front_image=front_png_b64, side_image=side_png_b64, height_cm=178)
    run(client, sid, 5, confirm=True)
    got = set(run(client, sid, 6)["measurements"])
    assert "inseam" in got
    assert "sleeve_length" not in got
    assert "chest" not in got


def test_export_yaml(client, front_png_b64, side_png_b64):
    sid = client.post("/api/session").json()["session_id"]
    run(client, sid, 1, garment_type="polo_shirt")
    run(client, sid, 2, front_image=front_png_b64, side_image=side_png_b64, height_cm=178)
    assert client.get(f"/api/session/{sid}/export.yaml").status_code == 400

    run(client, sid, 5, confirm=True)
    run(client, sid, 6)
    run(client, sid, 7)

    r = client.get(f"/api/session/{sid}/export.yaml")
    assert r.status_code == 200
    assert "attachment" in r.headers["content-disposition"]
    import yaml

    doc = yaml.safe_load(r.text)
    assert doc["session"]["garment_type"] == "polo_shirt"
    assert doc["measurements"]


def test_step8_route_toggle(client, front_png_b64, side_png_b64):
    sid = client.post("/api/session").json()["session_id"]
    run(client, sid, 1, garment_type="polo_shirt")
    run(client, sid, 2, front_image=front_png_b64, side_image=side_png_b64, height_cm=178)
    run(client, sid, 5, confirm=True)
    run(client, sid, 6)

    block = run(client, sid, 8, route="block")
    gc = run(client, sid, 8, route="garmentcode")
    assert block["route"] == "block"
    assert gc["route"] == "garmentcode"
    assert [p["name"] for p in block["pieces"]] == [p["name"] for p in gc["pieces"]]
    assert block["pieces"][0]["svg"] != gc["pieces"][0]["svg"]
    bad = client.post(f"/api/session/{sid}/step/8", json={"payload": {"route": "nope"}})
    assert bad.status_code == 400


def test_bad_requests_are_rejected(client, session_id):
    assert client.post(f"/api/session/{session_id}/step/1", json={}).status_code == 400
    assert (
        client.post(f"/api/session/{session_id}/step/1", json={"garment_type": "hat"}).status_code
        == 400
    )
    assert client.post(f"/api/session/{session_id}/step/2", json={}).status_code == 400
    assert client.post(f"/api/session/{session_id}/step/99", json={}).status_code == 404


def test_no_image_is_written_to_disk(client, front_png_b64, side_png_b64):
    """The acceptance criterion. Snapshot the repo tree, run everything, compare."""
    def tree() -> set[str]:
        out = set()
        for root, dirs, files in os.walk(REPO):
            dirs[:] = [d for d in dirs if d not in {".git", "node_modules", "__pycache__",
                                                    ".venv", "venv", ".pytest_cache", "dist"}]
            for f in files:
                out.add(os.path.relpath(os.path.join(root, f), REPO))
        return out

    before = tree()
    sid = client.post("/api/session").json()["session_id"]
    full_run(client, sid, "polo_shirt", front_png_b64, side_png_b64)
    full_run(client, sid, "slacks", front_png_b64, side_png_b64)

    added = tree() - before
    # nothing image-shaped, and nothing outside the cache dirs pytest just made
    offenders = [
        p for p in added
        if p.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"))
        and "__pycache__" not in p
    ]
    assert offenders == [], f"images were written to disk: {offenders}"
