import json
import threading
import urllib.error
import urllib.request

import pytest

import web
from pattern_drafter.measurements import MeasurementError

BODY_YAML = """
body:
  bust: 85.8
  waist: 85.6
  hips: 100.5
  arm_length: 62.6
  wrist: 16.2
"""


def payload(**overrides):
    body = {
        "measurements": BODY_YAML,
        "garment": "skirt",
        "style": "Gilewska",
        "gender": "w",
        "ease": 8,
        "darts": False,
    }
    body.update(overrides)
    return body


@pytest.fixture(scope="module")
def server():
    httpd = web.ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield "http://127.0.0.1:%d" % httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def request(url, data=None):
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


def test_index_served(server):
    status, body, headers = request(server + "/")

    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert b"body_measurements.yaml" in body


def test_garment_catalogue(server):
    status, body, _ = request(server + "/api/garments")

    catalogue = json.loads(body)
    assert status == 200
    assert [g["id"] for g in catalogue["garments"]] == ["bodice", "skirt", "trousers"]
    assert [v["gender"] for v in catalogue["garments"][-1]["variants"]] == ["m", "w"]
    assert {v["style"] for v in catalogue["garments"][-1]["variants"]} == {"Donnanno"}
    assert catalogue["garments"][0]["supports_darts"] is True


def test_draft_pattern_and_downloads(server):
    status, body, _ = request(
        server + "/api/pattern", json.dumps(payload()).encode("utf-8")
    )
    assert status == 200
    result = json.loads(body)

    assert result["template"] == "skirt"
    assert len(result["pattern"]["panels"]) == 2
    assert "tour_poitrine" in result["from_body"]
    assert result["reference"]

    for fmt in ("json", "pdf"):
        code, blob, headers = request(server + result["downloads"][fmt])
        assert code == 200
        assert headers["Content-Type"] == web.CONTENT_TYPES[fmt]
        assert len(blob) > 100
        assert fmt in headers["Content-Disposition"]


def test_bad_request_is_reported_not_raised(server):
    status, body, _ = request(
        server + "/api/pattern", json.dumps(payload(garment="hat")).encode("utf-8")
    )

    assert status == 400
    assert "unknown garment" in json.loads(body)["error"]


def test_invalid_json_body(server):
    status, body, _ = request(server + "/api/pattern", b"not json")

    assert status == 400
    assert "invalid JSON" in json.loads(body)["error"]


def test_unknown_routes_are_404(server):
    for path in ("/nope", "/download/deadbeef.pdf", "/api/nope"):
        status, _, _ = request(server + path, b"{}" if path == "/api/nope" else None)
        assert status == 404, path


def test_draft_validates_its_payload():
    with pytest.raises(MeasurementError):
        web.draft(payload(measurements="body:\n  bust: wide\n"))
    with pytest.raises(MeasurementError):
        web.draft(payload(ease="lots"))